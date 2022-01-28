"""
retina_rl library

"""

from torch import nn

from sample_factory.algorithms.appo.model_utils import register_custom_encoder, EncoderBase, get_obs_shape, nonlinearity
from sample_factory.algorithms.utils.pytorch_utils import calc_num_elements

### Simple Model ###

class SimpleEncoderBase(EncoderBase):

    def __init__(self, cfg, obs_space, timing):

        super().__init__(cfg, timing)

        obs_shape = get_obs_shape(obs_space)
        self.kernel_size = 3

        self.conv = nn.Conv2d(3, 16, 5, stride=1)

        self.nl = nonlinearity(cfg)

        # Preparing Fully Connected Layers
        conv_layers = [
            self.conv, self.nl
        ]

        self.conv_head = nn.Sequential(*conv_layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape.obs)

        self.encoder_out_size = cfg.hidden_size
        self.fc1 = nn.Linear(self.conv_head_out_size,self.encoder_out_size)

    def forward(self, x):
        # we always work with dictionary observations. Primary observation is available with the key 'obs'
        x = self.nl(self.conv(x))
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.nl(self.fc1(x))
        return x

class SimpleEncoder(SimpleEncoderBase):

    def __init__(self, cfg, obs_space,timing):

        super().__init__(cfg,obs_space,timing)
        self.base_encoder = SimpleEncoderBase(cfg,obs_space,timing)

    def forward(self, obs_dict):
        # we always work with dictionary observations. Primary observation is available with the key 'obs'
        main_obs = obs_dict['obs']

        # forward pass through configurable fully connected blocks immediately after the encoder
        x = self.base_encoder(main_obs)
        return x

### Retinal-VVS Model ###

class LindseyEncoderBase(EncoderBase):

    def __init__(self, cfg, obs_space, timing):

        super().__init__(cfg, timing)

        nchns = cfg.global_channels
        btlchns = cfg.retinal_bottleneck
        vvsdpth = cfg.vvs_depth
        krnsz = cfg.kernel_size

        self.nl_fc = nonlinearity(cfg)
        self.kernel_size = krnsz

        # Preparing Conv Layers
        conv_layers = []
        self.nls = []
        for i in range(vvsdpth+2): # +2 for the first 'retinal' layers
            self.nls.append(nonlinearity(cfg))
            if i == 0: # 'bipolar cells' ('global channels')
                conv_layers.extend([nn.Conv2d(3, nchns, krnsz, stride=1), self.nls[i]])
            elif i == 1: # 'ganglion cells' ('retinal bottleneck')
                conv_layers.extend([nn.Conv2d(nchns, btlchns, krnsz, stride=1), self.nls[i]])
            elif i == 2: # 'V1' ('global channels')
                conv_layers.extend([nn.Conv2d(btlchns, nchns, krnsz, stride=1), self.nls[i]])
            else: # 'vvs layers'
                conv_layers.extend([nn.Conv2d(nchns, nchns, krnsz, stride=1), self.nls[i]])

        self.conv_head = nn.Sequential(*conv_layers)
        obs_shape = get_obs_shape(obs_space)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape.obs)
        self.encoder_out_size = cfg.hidden_size
        self.fc1 = nn.Linear(self.conv_head_out_size,self.encoder_out_size)

    def forward(self, x):
        # we always work with dictionary observations. Primary observation is available with the key 'obs'
        x = self.conv_head(x)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.nl_fc(self.fc1(x))
        return x

class LindseyEncoder(LindseyEncoderBase):

    def __init__(self, cfg, obs_space,timing):

        super().__init__(cfg,obs_space,timing)
        self.base_encoder = LindseyEncoderBase(cfg,obs_space,timing)

    def forward(self, obs_dict):
        # we always work with dictionary observations. Primary observation is available with the key 'obs'
        main_obs = obs_dict['obs']

        # forward pass through configurable fully connected blocks immediately after the encoder
        x = self.base_encoder(main_obs)
        return x

### MosaicRetinal-VVS Model ###

class MosaicEncoderBase(EncoderBase):

    def __init__(self, cfg, obs_space, timing):

        super().__init__(cfg, timing)

        nchns = cfg.global_channels
        btlchns = cfg.retinal_bottleneck
        vvsdpth = cfg.vvs_depth

        # bc layer architecture
        krnsz1 = cfg.kernel_size
        strd1 = krnsz1 # mosaic in conv1 (stride has to be the same as kernel)
        
        # rgc layer architecture
        krnsz2 = cfg.rf_ratio # 'how many bcs will one rgc pool'
        strd2 = krnsz2
        
        # v1/vvs layer architecture
        krnszv = cfg.rf_ratio # TODO: think of good ratio for V1
        strdv = 1 # there is no mosaic in V1 or higher areas
        
        self.nl_fc = nonlinearity(cfg)
        self.kernel_size = krnsz1 # might be redundant also in lindsey implementation

        # rf sizes/diameters (wrt pixels) - ised for plotting

        # Preparing Conv Layers
        conv_layers = []
        self.nls = []
        for i in range(vvsdpth+2): # +2 for the first 'retinal' layers
            self.nls.append(nonlinearity(cfg))
            if i == 0: # 'bipolar cells' ('global channels')
                conv_layers.extend([nn.Conv2d(3, nchns, krnsz1, stride=strd1), self.nls[i]])
            elif i == 1: # 'ganglion cells' ('retinal bottleneck')
                conv_layers.extend([nn.Conv2d(nchns, btlchns, krnsz2, stride=strd2), self.nls[i]])
            elif i == 2: # 'V1' ('global channels')
                conv_layers.extend([nn.Conv2d(btlchns, nchns, krnszv, stride=strdv), self.nls[i]])
            else: # 'vvs layers'
                conv_layers.extend([nn.Conv2d(nchns, nchns, krnszv, stride=strdv), self.nls[i]])

        self.conv_head = nn.Sequential(*conv_layers)
        obs_shape = get_obs_shape(obs_space)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape.obs)
        self.encoder_out_size = cfg.hidden_size
        self.fc1 = nn.Linear(self.conv_head_out_size,self.encoder_out_size)

    def forward(self, x):
        # we always work with dictionary observations. Primary observation is available with the key 'obs'
        x = self.conv_head(x)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.nl_fc(self.fc1(x))
        return x

class MosaicEncoder(MosaicEncoderBase):

    def __init__(self, cfg, obs_space,timing):

        super().__init__(cfg,obs_space,timing)
        self.base_encoder = MosaicEncoderBase(cfg,obs_space,timing)

    def forward(self, obs_dict):
        # we always work with dictionary observations. Primary observation is available with the key 'obs'
        main_obs = obs_dict['obs']

        # forward pass through configurable fully connected blocks immediately after the encoder
        x = self.base_encoder(main_obs)
        return x

### Linear encoder ('negative control') ###

class LinearEncoderBase(EncoderBase):

    def __init__(self, cfg, obs_space, timing):

        super().__init__(cfg, timing)

        # to get input size
        obs_shape = get_obs_shape(obs_space)
        self.input_flat_shape = obs_shape.obs[0]* obs_shape.obs[1]* obs_shape.obs[2] # c * h * w

        self.nl = nonlinearity(cfg) # for pixels this doesn't matter as long as it's (r)elu

        self.encoder_out_size = cfg.hidden_size
        self.fc1 = nn.Linear(self.input_flat_shape,self.encoder_out_size)

    def forward(self, x):
        x = x.contiguous().view(-1, self.input_flat_shape)
        x = self.nl(self.fc1(x))
        return x

class LinearEncoder(LinearEncoderBase):

    def __init__(self, cfg, obs_space,timing):

        super().__init__(cfg,obs_space,timing)
        self.base_encoder = LinearEncoderBase(cfg,obs_space,timing)

    def forward(self, obs_dict):
        # we always work with dictionary observations. Primary observation is available with the key 'obs'
        main_obs = obs_dict['obs']

        # forward pass through configurable fully connected blocks immediately after the encoder
        x = self.base_encoder(main_obs)
        return x


def register_encoders():
    register_custom_encoder('lindsey', LindseyEncoder)
    register_custom_encoder('simple', SimpleEncoder)
    register_custom_encoder('linear', LinearEncoder)
    register_custom_encoder('mosaic', MosaicEncoder)
