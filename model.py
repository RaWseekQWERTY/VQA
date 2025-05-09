import torch.nn as nn
import torch

class VQAModel(nn.Module):
    def __init__(self, vit_model, bert_model, hidden_dim, num_classes):
        super(VQAModel, self).__init__()
        self.vit = vit_model
        self.bert = bert_model

        self.image_dim = vit_model.config.hidden_size
        self.text_dim = bert_model.config.hidden_size
        self.hidden_dim = hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(self.image_dim + self.text_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        image_features = self.vit(pixel_values=image).last_hidden_state[:, 0, :]
        text_features = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        fused = torch.cat((image_features, text_features), dim=1)
        return self.fc(fused)
