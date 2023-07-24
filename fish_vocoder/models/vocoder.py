import lightning as L


class VocoderModel(L.LightningModule):
    def visualize(self, x):
        return self(x)
