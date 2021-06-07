from collections import namedtuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from absl import app, flags
from .attention import SimpleAttention
from .utils import weight_top_p, trim
from .projection import SoftAlign
import pdb
#_VF = torch._C._VariableFunctions
EPS = 1e-7
FLAGS = flags.FLAGS

DecoderState = namedtuple("DecoderState", "feed rnn_state hiddens tokens")
BeamState = namedtuple("BeamState", "feed rnn_state hiddens tokens score parent done")

class Decoder(nn.Module):
    def __init__(
            self,
            vocab,
            n_embed,
            n_hidden,
            n_layers,
            attention=None,
            copy=False,
            self_attention=False,
            dropout=0,
            rnntype=nn.LSTM,
            concat_feed=True,
            MAXLEN=45,
    ):
        super().__init__()

        # setup
        self.vocab = vocab
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.copy = copy
        self.self_attention = self_attention
        self.rnntype=rnntype
        self.MAXLEN = MAXLEN
        # attention
        if attention is None:
            attention = ()
        attention = tuple(attention)
        if self_attention:
            attention = attention + (SimpleAttention(n_hidden, n_hidden),)
        self.attention = attention
        for i, att in enumerate(attention):
            self.add_module("attention_%d" % i, att)
        # modules
        self.embed = nn.Embedding(len(vocab), n_embed, vocab.pad())

        self.dropout_in = nn.Dropout(dropout)
        self.predict = nn.Linear(n_hidden, len(vocab))
        #self.copy_switch = nn.Linear(n_hidden, 1 + len(attention))
        self.concat_feed = concat_feed
        if self.concat_feed:
            self.combine = nn.Linear(n_embed + n_hidden * (2 + len(attention)), n_hidden)
            self.rnn = self.rnntype(n_embed + n_hidden, n_hidden, n_layers, dropout=dropout)
        else:
            self.combine = nn.Linear(n_embed + n_hidden * (1 + len(attention)), n_hidden)
            self.rnn = self.rnntype(n_embed, n_hidden, n_layers, dropout=dropout)

        self.dropout_out = nn.Dropout(dropout)
        self.seq_picker = nn.Linear(n_hidden, len(attention))
        self.nll = nn.CrossEntropyLoss(reduction='none', ignore_index=vocab.pad())
        self.nllreduce = nn.CrossEntropyLoss(ignore_index=vocab.pad())
        self.copy_translation =  None
        self.high_drop = nn.Dropout(FLAGS.highdropvalue)

    def step(
            self,
            decoder_state,
            att_features,
            att_tokens,
            att_masks,
            att_token_proj,
            self_att_proj
    ):
        # advance rnn
        if decoder_state.tokens.dtype == torch.float32:
            emb = torch.matmul(decoder_state.tokens, self.embed.weight)
        else:
            emb = self.embed(decoder_state.tokens[-1, :])

        if self.concat_feed:
            inp = self.dropout_in(torch.cat((emb, decoder_state.feed), dim=1))
        else:
            inp = self.dropout_in(emb)


        hidden, rnn_state = self.rnn(inp.unsqueeze(0), decoder_state.rnn_state)

        # prep self-attention
        if self.self_attention:
            hiddens = torch.cat(decoder_state.hiddens + [hidden], dim=0)
            att_features = tuple(att_features) + (hiddens,)
            att_tokens = tuple(att_tokens) + (decoder_state.tokens,)
            att_masks = att_masks + (
                (decoder_state.tokens == self.vocab.pad()).float(),
            )
            att_token_proj = att_token_proj + (self_att_proj,)

        # advance attention
        attended = [
            attention(features, hidden, mask)
            for attention, features, mask in zip(
                self.attention, att_features, att_masks
            )
        ]
        if len(attended) > 0:
            summary, distribution, copy_dist = zip(*attended)
        else:
            summary = distribution = copy_dist = ()

        all_features = torch.cat([inp.unsqueeze(0),hidden] + list(summary), dim=-1)

        comb_features = self.dropout_out(self.combine(all_features).squeeze(0))

        if FLAGS.highdrop:
            if FLAGS.highdroptest:
                comb_features = F.dropout(comb_features,p=self.high_drop.p,training=True)
            else:
                comb_features = self.high_drop(comb_features)
            #comb_features = F.dropout(comb_features,p=0.5,training=True)

        pred_logits = self.predict(comb_features) # REQUIRES FOR JUMP

        assert not torch.isnan(pred_logits).any()

        if self.copy:
            dists = copy_dist #distribution
            if len(dists) > 1:
                seq_probs = self.seq_picker(hidden).softmax(dim=2)
                weighted_dists = [dists[i] * seq_probs[:, :, i] for i in range(len(dists))]
            else:
                weighted_dists = dists

            pred_probs = F.softmax(pred_logits, dim=1)
            copy_weights = pred_probs[:, self.vocab.copy()].unsqueeze(-1)
            projs = att_token_proj
            copy_probs = [(weighted_dists[i].unsqueeze(2) * projs[i]).sum(0) for i in range(len(dists))]
            copy_probs = sum(copy_probs) * copy_weights
            # pdb.set_trace()
            copy_probs += EPS
            comb_probs  = copy_probs + pred_probs

            # copy_mask   = torch.ones_like(comb_probs)
            # copy_mask[:, self.vocab.copy()] = EPS
            # comb_probs = comb_probs * copy_mask
            comb_probs[:, self.vocab.copy()] = EPS

            direct_logits = pred_logits
            copy_logits = torch.log(copy_probs)
            pred_logits = torch.log(comb_probs)
        else:
            direct_logits  = pred_logits
            copy_logits    = None
            weighted_dists = None
            dists = None

        # done
        return (
            pred_logits,
            comb_features,
            rnn_state,
            hidden,
            direct_logits, copy_logits, dists
        )

    def _make_projection(self, tokens):
        if self.copy:
            if type(self.copy_translation) == SoftAlign:
                return self.copy_translation(tokens)
            else:
                proj = np.zeros(
                    (tokens.shape[0], tokens.shape[1], len(self.vocab)), dtype=np.int64
                )
                device = tokens.device
                tokens = tokens.cpu().numpy()
                for i in range(tokens.shape[0]):
                    if self.copy_translation is None:
                        proj[i, range(tokens.shape[1]), tokens[i, :]] = 1
                    else:
                        proj[i, range(tokens.shape[1]), self.copy_translation[tokens[i, :]]] = 1
                return torch.tensor(proj, dtype=torch.float32, device=device)
        else:
            return None


    def get_mask(self, toks):
        if toks.dtype == torch.float32:
            mask = (torch.argmax(toks.detach(),dim=-1) == self.vocab.pad()).float()
        else:
            mask = (toks == self.vocab.pad()).float()
        return mask


    def forward(
            self,
            rnn_state,
            max_len,
            ref_tokens=None,
            att_features=None,
            att_tokens=None,
            token_picker=None
    ):
        # token picker
        sampling = True
        if token_picker is None:
            sampling = False
            if self.self_attention:
                master_self_att_proj = self._make_projection(ref_tokens)
            def token_picker(t, logits):
                if t < ref_tokens.shape[0]:
                    if self.self_attention:
                        return ref_tokens[t, :], master_self_att_proj[:t+1, :, :]
                    else:
                        return ref_tokens[t, :], None
                else:
                    return None, None


        # attention
        if att_features is None:
            att_features = ()
            att_tokens = ()
            att_masks = ()
            att_token_proj = ()
        else:
            assert isinstance(att_features, list) \
                or isinstance(att_features, tuple)
            assert len(att_features) == len(att_tokens)
            assert len(self.attention) == len(att_features) + (1 if self.self_attention else 0)
            att_masks = tuple(
                self.get_mask(toks) for toks in att_tokens
            )
            att_token_proj = tuple(
                self._make_projection(toks) for toks in att_tokens
            )

        # init
        pred = None
        tokens, self_att_proj = token_picker(0, pred)
        feed = tokens.new_zeros(
            tokens.shape[0], self.n_hidden
        ).float()
        hiddens = []
        all_tokens = [tokens]
        all_preds = []
        all_extra = []

        if FLAGS.debug:
            print("max_len: ",max_len)
        for t in range(max_len):
            decoder_state = DecoderState(
                feed, rnn_state, hiddens, torch.stack(all_tokens)
            )

            pred, feed, rnn_state, hidden, *extra = self.step(
                decoder_state,
                att_features,
                att_tokens,
                att_masks,
                att_token_proj,
                self_att_proj,
            )

            if FLAGS.debug:
                pdb.set_trace()

            new_pred = torch.zeros_like(pred)
            new_pred[:, self.vocab.pad():self.vocab.unk()+1]  -= 99999
            new_pred[:, self.vocab.eos()] += 99999 # put eos back TODO single step
            pred = pred + new_pred

            if  t == self.MAXLEN-1: #FIXME: when sampling there is one more extra timestep
                new_pred = torch.zeros_like(pred)
                new_pred[:, self.vocab.eos()] += 99999
                pred = pred + new_pred

            if FLAGS.debug:
                pdb.set_trace()

            hiddens.append(hidden)
            all_preds.append(pred)
            all_extra.append(extra)

            tokens, self_att_proj = token_picker(t+1, pred)

            if tokens is None:
                break

            all_tokens.append(tokens)

        return (
            torch.stack(all_preds),
            torch.stack(all_tokens),
            rnn_state,
            list(zip(*all_extra))
        )

    def sample(
            self,
            rnn_state,
            max_len,
            n_batch=1,
            temp=1.0,
            att_features=None,
            att_tokens=None,
            greedy=False,
            top_p=None,
            custom_sampler=None,
            calc_score=False,
    ):

        device = self.embed.weight.device

        # init
        if self.rnntype == nn.LSTM:
            if rnn_state is None:
                rnnstate = tuple(torch.zeros(self.n_layers, n_batch, self.n_hidden).to(device) for _ in range(2))
            else:
                n_batch = rnn_state[0].shape[1]
        else:
            if rnn_state is None:
                rnnstate = torch.zeros(self.n_layers, n_batch, self.n_hidden).to(device)
            else:
                n_batch = rnn_state.shape[1]

        done = np.array([False] * n_batch)
        running_proj = torch.zeros(max_len+1, n_batch, len(self.vocab)).to(device)
        running_out = torch.zeros(max_len+1, n_batch, dtype=torch.int64).to(device)

        def token_picker(t, logits):
            # first step
            sos, eos, pad = self.vocab.sos(), self.vocab.eos(), self.vocab.pad()
            if t == 0:
                toks = torch.LongTensor([sos] * n_batch).to(device)
                running_proj[0, range(n_batch), toks] = 1
                running_out[0, range(n_batch)] = 1
                return toks, running_proj[:1, :, :]
            if all(done):
                return None, None

            new_pred = torch.zeros_like(logits)
            new_pred[np.flatnonzero(done), pad] += 2*99999
            logits = (logits + new_pred).detach()

            # pdb.set_trace()
            if greedy:
                toks = torch.argmax(logits, dim=1)
                done[torch.eq(toks,eos).cpu().numpy()] = True
            elif custom_sampler is None:
                logits = logits / temp
                if top_p is not None:
                    top_values, top_indices = torch.topk(logits, top_p, dim=1)
                    kth_best = top_values[:, -1].view([-1, 1])
                    kth_best = kth_best.repeat([1, logits.shape[1]]).float()
                    ignore = torch.lt(logits, kth_best)
                    logits = logits.masked_fill(ignore, -99999)

                dist = torch.distributions.Multinomial(logits=logits, total_count=1)
                toks = torch.argmax(dist.sample(), dim=-1)
                done[torch.eq(toks,eos).cpu().numpy()] = True
            else:
                probs = F.softmax(logits / temp, dim=-1)
                probs = probs.detach().cpu().numpy()
                tokens = []
                for i, row in enumerate(probs):
                    if done[i]:
                        tokens.append(self.vocab.pad())
                        continue
                    if custom_sampler is not None:
                        row = torch.tensor(row).unsqueeze(0)
                        choice = custom_sampler(row, running_out[:t])
                    tokens.append(choice)
                    if choice == self.vocab.eos():
                        done[i] = True
                toks = torch.LongTensor(tokens).to(device)

            running_proj[t, range(n_batch), toks] = 1
            running_out[t, range(n_batch)] = toks
            return toks, running_proj[:t+1, :, :]

        preds, tokens, rnn_state, *_ = self(
            rnn_state,
            max_len,
            att_features=att_features,
            att_tokens=att_tokens,
            token_picker=token_picker
        )

        tok_arr = tokens.detach().cpu().numpy().transpose()[:,1:]
        tok_out = []
        if calc_score:
            preds = F.log_softmax(preds, dim=-1)
            score_out = [0] * tok_arr.shape[0]
            for i, row in enumerate(tok_arr):
                row_out = []
                for t, c in enumerate(row):
                    row_out.append(int(c))
                    if c != self.vocab.pad():
                        score_out[i] += preds[t, i, c].item()
                    if c == self.vocab.eos():
                        break
                tok_out.append([self.vocab.sos()]+row_out)
        else:
            tok_out = [[self.vocab.sos()]+trim(row.tolist(),self.vocab.eos()) for i, row in enumerate(tok_arr)]
            score_out = None
        return tok_out, score_out


    def sample_with_gumbel(self,
                           rnn_state,
                           max_len,
                           n_batch=1,
                           att_features=None,
                           att_tokens=None,
                           temp=1.0,
                           calc_score=False):

        device = self.embed.weight.device
        # init
        if self.rnntype == nn.LSTM:
            if rnn_state == None:
                rnnstate = tuple(torch.zeros(self.n_layers, n_batch, self.n_hidden).to(device) for _ in range(2))
            else:
                n_batch = rnn_state[0].shape[1]
        else:
            if rnn_state == None:
                rnnstate = torch.zeros(self.n_layers, n_batch, self.n_hidden).to(device)
            else:
                n_batch = rnn_state.shape[1]

        done = np.array([False] * n_batch)
        running_proj = torch.zeros(max_len+1, n_batch, len(self.vocab)).to(device)
        running_out  = torch.zeros(max_len+1, n_batch, dtype=torch.int64).to(device)

        def token_picker(t, logits):
            sos, eos, pad = self.vocab.sos(), self.vocab.eos(), self.vocab.pad()
            if t == 0:
                toks = torch.LongTensor([sos] * n_batch).to(device)
                onehots = F.one_hot(toks, num_classes=len(self.vocab)).float()
                running_proj[0, range(n_batch), toks] = 1
                running_out[0, range(n_batch)] = sos
                return toks, running_proj[:1, :, :], onehots

            if all(done) or t == self.MAXLEN:
                return None, None, None

            new_pred = torch.zeros_like(logits)
            new_pred[np.flatnonzero(done), pad] += 2*99999
            logits = logits + new_pred

            onehots = F.gumbel_softmax(logits, tau=temp, hard=True, dim=-1)
            toks = torch.argmax(onehots.detach(), dim=1)
            done[torch.eq(toks,eos).cpu().numpy()] = True
            # pdb.set_trace()

            running_proj[t, range(n_batch), toks] = 1
            running_out[t, range(n_batch)] = toks
            return toks, running_proj[:t+1, :, :], onehots

        preds, tokens, rnn_state, *_ = self.forward_onehot(rnn_state,
            max_len,
            att_features=att_features,
            att_tokens=att_tokens,
            token_picker=token_picker
        )


        if calc_score:
            logprob = (preds * tokens).sum(dim=-1).sum(dim=0)
        else:
            logprob = None

        return tokens, logprob

    def forward_onehot(self,
                       rnn_state,
                       max_len,
                       ref_tokens=None,
                       att_features=None,
                       att_tokens=None,
                       token_picker=None):

        sampling = True
        if token_picker is None:
            sampling = False
            master_self_att_proj = None
            def token_picker(t, logits):
                if t < ref_tokens.shape[0]:
                    onehots = ref_tokens[t, :, :]
                    tokens  = torch.argmax(onehots.detach(),dim=-1)
                    return tokens, None, onehots
                else:
                    return None, None, None


        # attention
        if att_features is None:
            att_features = ()
            att_tokens = ()
            att_masks = ()
            att_token_proj = ()
        else:
            assert isinstance(att_features, list) \
                or isinstance(att_features, tuple)
            assert len(att_features) == len(att_tokens)
            assert len(self.attention) == len(att_features) + (1 if self.self_attention else 0)
            att_masks = tuple(
                (toks == self.vocab.pad()).float() for toks in att_tokens
            )
            att_token_proj = tuple(
                self._make_projection(toks) for toks in att_tokens
            )

        # init
        pred = None
        tokens, self_att_proj , onehots = token_picker(0, pred)
        feed = tokens.new_zeros(
            tokens.shape[0], self.n_hidden
        ).float()

        hiddens = []
        all_tokens = [tokens]
        all_tokens_onehot = [onehots]
        all_preds = []
        all_extra = []

        # iter
        if FLAGS.debug:
            print("max_len: ",max_len)

        for t in range(max_len):
            decoder_state = DecoderState(
                feed, rnn_state, hiddens, onehots
            )

            pred, feed, rnn_state, hidden, *extra = self.step(
                decoder_state,
                att_features,
                att_tokens,
                att_masks,
                att_token_proj,
                self_att_proj,
            )

            new_pred = torch.zeros_like(pred)
            new_pred[:, self.vocab.pad():self.vocab.unk()+1]  -= 99999
            new_pred[:, self.vocab.eos()] += 99999 # put eos back
            pred = pred + new_pred

            if  t == self.MAXLEN-1: #FIXME: when sampling there is one more extra timestep
                new_pred = torch.zeros_like(pred)
                new_pred[:, self.vocab.eos()] += 99999
                pred = pred + new_pred
#                 pred[:,self.vocab.eos()] += 99999


            hiddens.append(hidden)
            all_preds.append(pred)
            all_extra.append(extra)

            tokens, self_att_proj, onehots = token_picker(t+1, pred)

            if tokens is None:
                break

            all_tokens.append(tokens)
            all_tokens_onehot.append(onehots)

        return (
            torch.stack(all_preds),
            torch.stack(all_tokens_onehot, dim=0),
            rnn_state,
            list(zip(*all_extra))
        )

    def logprob(self, ref_tokens, rnn_state=None, att_features=None, att_tokens=None):
        device = self.embed.weight.device
        n_batch = ref_tokens.shape[1]
        if self.rnntype == nn.LSTM and rnn_state is None:
                rnnstate = tuple(torch.zeros(self.n_layers, n_batch, self.n_hidden).to(device) for _ in range(2))
        elif rnn_state is None:
                rnnstate = torch.zeros(self.n_layers, n_batch, self.n_hidden).to(device)

        if len(ref_tokens.shape) == 3:
            out_src = ref_tokens[:-1, :, :]
            preds, _, _, *_ = self.forward_onehot(rnn_state,
                                                  out_src.shape[0],
                                                  ref_tokens=out_src,
                                                  att_features=att_features,
                                                  att_tokens=att_tokens)
            out_tgt = ref_tokens[1:,:,:]
            ids = torch.argmax(out_tgt.detach(),dim=-1)
            out_mask = (ids != self.vocab.pad()).unsqueeze(2).float()
            logp = F.log_softmax(preds,dim=-1)
            logprob = (logp * (out_tgt * out_mask)).sum(dim=-1).sum(dim=0)
        else:
            out_src = ref_tokens[:-1, :]
            preds, _, _, *_ = self.forward(rnn_state,
                                           out_src.shape[0],
                                           ref_tokens=out_src,
                                           att_features=att_features,
                                           att_tokens=att_tokens)
            logp = F.log_softmax(preds,dim=-1)
            if FLAGS.debug:
                print("preds here")
                pdb.set_trace()
            out_tgt = ref_tokens[1:,:]
            logprob = -self.nll(logp.permute(1,2,0), out_tgt.transpose(0,1)).sum(dim=-1)
        return logprob


    def beam(
            self,
            rnn_state,
            beam_size,
            max_len,
            att_features=None,
            att_tokens=None,
    ):
        assert rnn_state[0].shape[1] == 1
        device = rnn_state[0].device

        # init attention
        if att_features is None:
            att_features = ()
            att_tokens = ()
            att_masks = ()
            att_token_proj = ()
        else:
            assert isinstance(att_features, list) \
                or isinstance(att_features, tuple)
            att_masks = tuple(
                (toks == self.vocab.pad()).float() for toks in att_tokens
            )
            att_token_proj = tuple(
                self._make_projection(toks) for toks in att_tokens
            )

        # initialize beam
        beam = [BeamState(
            rnn_state[0].new_zeros(self.n_hidden),
            [s.squeeze(1) for s in rnn_state],
            [],
            [self.vocab.sos()],
            0.,
            None,
            False
        )]

        for t in range(max_len):
            if all(s.done for s in beam):
                break
            rnn_state = [
                torch.stack([s.rnn_state[i] for s in beam], dim=1)
                for i in range(len(beam[0].rnn_state))
            ]
            tokens = torch.LongTensor([
                [s.tokens[tt] if tt < len(s.tokens) else s.tokens[-1] for s in beam]
                for tt in range(t+1)
            ]).to(device)
            decoder_state = DecoderState(
                torch.stack([s.feed for s in beam]),
                rnn_state,
                [torch.stack(
                    [s.hiddens[tt] if tt < len(s.hiddens) else s.hiddens[-1] for s in beam],
                dim=1) for tt in range(t)],
                tokens,
            )
            self_att_proj = self._make_projection(tokens)
            pred, feed, rnn_state, hidden, *_ = self.step(
                decoder_state,
                tuple(f.expand(f.shape[0], len(beam), f.shape[2]) for f in att_features),
                tuple(t.expand(t.shape[0], len(beam)) for t in att_tokens),
                tuple(m.expand(m.shape[0], len(beam)) for m in att_masks),
                tuple(p.expand(p.shape[0], len(beam), p.shape[2]) for p in att_token_proj if p is not None),
                self_att_proj
            )

            logprobs = F.log_softmax(pred, dim=1)
            next_beam = []
            for i, row in enumerate(logprobs):
                row[self.vocab.copy()] = -np.inf
                scores, toks = row.topk(beam_size)
                if beam[i].done:
                    next_beam.append(beam[i])
                else:
                    for s, t in zip(scores, toks):
                        next_beam.append(BeamState(
                            feed[i, :],
                            [s[:, i, :] for s in rnn_state],
                            beam[i].hiddens + [hidden[:, i, :]],
                            beam[i].tokens + [t.item()],
                            beam[i].score + s,
                            beam[i],
                            t == self.vocab.eos()
                        ))
            next_beam = sorted(next_beam, key=lambda x: -x.score)
            beam = next_beam[:beam_size]

        return [s.tokens for s in beam]
