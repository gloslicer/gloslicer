import torch
import torch.nn as nn
from transformers import RobertaModel, LongformerModel, AutoConfig
import os
import logging

logger = logging.getLogger(__name__)

class SliceMLP(nn.Module):
    '''Backward/Forward slicing classification head.'''
    def __init__(self, config):
        super(SliceMLP, self).__init__()
        self.fc1 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc3 = nn.Linear(config.hidden_size, 1)
        self.forward_activation = torch.nn.GELU()
        classifier_dropout = getattr(config, "classifier_dropout", None)
        if classifier_dropout is None:
            classifier_dropout = getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(classifier_dropout)

    def forward(self, x):
        x = self.forward_activation(self.fc1(x))
        x = self.dropout(x)
        x = self.forward_activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        # No sigmoid here!
        return x

class AutoSlicingModel(nn.Module):
    def __init__(self, args, config, pos_weight_back=None, pos_weight_forward=None):
        super(AutoSlicingModel, self).__init__()
        self.max_tokens = args.max_tokens
        self.use_statement_ids = args.use_statement_ids
        self.pooling_strategy = args.pooling_strategy

        # Encoder loading logic unchanged
        model_path = None
        if args.model_key == "allenai/longformer-base-4096":
            model_path = "./model/base_model/longformer-base-4096"
        elif args.model_key == "allenai/longformer-large-4096":
            model_path = "./model/base_model/longformer-large-4096"
        elif args.model_key == "microsoft/codebert-base":
            model_path = "./model/base_model/codebert-base"
        elif args.model_key == "microsoft/graphcodebert-base":
            model_path = "./model/base_model/graphcodebert-base"

        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            if 'longformer' in args.model_key:
                self.encoder = LongformerModel.from_pretrained(model_path, config=config)
            else:
                self.encoder = RobertaModel.from_pretrained(model_path, config=config)
        else:
            logger.info(f"Loading model from {args.model_key}")
            if 'longformer' in args.model_key:
                self.encoder = LongformerModel.from_pretrained(args.model_key, config=config)
            else:
                self.encoder = RobertaModel.from_pretrained(args.model_key, config=config)

        for param in self.encoder.parameters():
            param.requires_grad = not args.pretrain

        if self.use_statement_ids:
            self.statement_embeddings = nn.Embedding(args.max_tokens, config.hidden_size)

        self.back_mlp = SliceMLP(config)
        self.forward_mlp = SliceMLP(config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if pos_weight_back is None:
            pos_weight_back = torch.tensor([1.0], dtype=torch.float32, device=device)
        if pos_weight_forward is None:
            pos_weight_forward = torch.tensor([1.0], dtype=torch.float32, device=device)

        self.loss_criterion_back = nn.BCEWithLogitsLoss(pos_weight=pos_weight_back, reduction='mean')
        self.loss_criterion_forward = nn.BCEWithLogitsLoss(pos_weight=pos_weight_forward, reduction='mean')

    def forward(self, inputs_ids, inputs_masks, statements_ids, variables_ids,
                variables_line_numbers, slices_labels=None, return_embeddings=False):
        device = inputs_ids.device if inputs_ids is not None else 'cpu'
        inputs_embeddings = self.encoder.embeddings.word_embeddings(inputs_ids)

        if self.use_statement_ids:
            inputs_embeddings += self.statement_embeddings(statements_ids)

        encoder_outputs = self.encoder(
            inputs_embeds=inputs_embeddings,
            attention_mask=inputs_masks,
            output_attentions=False,
            output_hidden_states=True,
        )

        hidden_states = encoder_outputs.hidden_states
        outputs_embeddings = hidden_states[-1]

        batch_preds = {'back': [], 'forward': []}
        batch_true = {'back': [], 'forward': []}
        batch_loss = torch.tensor(0, dtype=torch.float, device=device)

        all_embeddings = []
        all_labels = []

        for _id, output_embeddings in enumerate(outputs_embeddings):
            statements_embeddings = []
            variable_ids = variables_ids[_id][torch.ne(variables_ids[_id], -1)]
            variable_toks_embeddings = output_embeddings[variable_ids]
            if self.pooling_strategy == 'mean':
                variable_embedding = torch.mean(variable_toks_embeddings, dim=0)
            elif self.pooling_strategy == 'max':
                variable_embedding = torch.max(variable_toks_embeddings, dim=0).values

            item_statements_ids = statements_ids[_id][torch.ne(statements_ids[_id], self.max_tokens - 1)]
            num_statements_in_item = torch.max(item_statements_ids).item()
            for sid in range(num_statements_in_item + 1):
                _statement_ids = (item_statements_ids == sid).nonzero().squeeze()
                if _statement_ids.dim() == 0:
                    _statement_ids = _statement_ids.unsqueeze(0)
                statement_toks_embeddings = output_embeddings[_statement_ids]
                if statement_toks_embeddings.shape[0] == 0:
                    continue
                if statement_toks_embeddings.dim() == 1:
                    statement_toks_embeddings = statement_toks_embeddings.unsqueeze(0)
                if self.pooling_strategy == 'mean':
                    statement_embedding = torch.mean(statement_toks_embeddings, dim=0)
                elif self.pooling_strategy == 'max':
                    statement_embedding = torch.max(statement_toks_embeddings, dim=0).values
                statements_embeddings.append(statement_embedding)

            back_statements_embeddings = statements_embeddings[:variables_line_numbers[_id]]
            forward_statements_embeddings = statements_embeddings[variables_line_numbers[_id] + 1:]

            device = output_embeddings.device
            empty_tensor = torch.tensor([], device=device)

            if len(back_statements_embeddings) == 0:
                batch_preds['back'].append(empty_tensor)
                if slices_labels is not None:
                    batch_true['back'].append(empty_tensor)
            else:
                preds_back = self.back_mlp(
                    torch.stack([torch.cat((x, variable_embedding))
                                for x in back_statements_embeddings])
                ).squeeze(-1)
                batch_preds['back'].append(preds_back)
                if slices_labels is not None:
                    item_slice_labels = slices_labels[_id][slices_labels[_id] != -1]
                    true_back = item_slice_labels[:variables_line_numbers[_id]]
                    batch_true['back'].append(true_back)
                    item_back_loss = self.loss_criterion_back(preds_back, true_back)
                    batch_loss += item_back_loss

            # forward
            if len(forward_statements_embeddings) == 0:
                batch_preds['forward'].append(empty_tensor)
                if slices_labels is not None:
                    batch_true['forward'].append(empty_tensor)
            else:
                preds_forward = self.forward_mlp(
                    torch.stack([torch.cat((x, variable_embedding))
                                for x in forward_statements_embeddings])
                ).squeeze(-1)
                batch_preds['forward'].append(preds_forward)
                if slices_labels is not None:
                    item_slice_labels = slices_labels[_id][slices_labels[_id] != -1]
                    true_forward = item_slice_labels[variables_line_numbers[_id] + 1:]
                    batch_true['forward'].append(true_forward)
                    item_forward_loss = self.loss_criterion_forward(preds_forward, true_forward)
                    batch_loss += item_forward_loss

            if return_embeddings:
                all_embeddings.extend(statements_embeddings)
                if slices_labels is not None:
                    item_slice_labels = slices_labels[_id][slices_labels[_id] != -1]
                    all_labels.extend(item_slice_labels.tolist())

        if return_embeddings:
            return all_embeddings, all_labels

        if slices_labels is None:
            return batch_preds
        else:
            return batch_loss, batch_preds, batch_true
