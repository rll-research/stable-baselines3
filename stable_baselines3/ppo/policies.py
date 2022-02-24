# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    MultiInputActorCriticPolicy,
    register_policy,
)

from stable_baselines3.common.policies_recurrent import (
    RecurrentActorCriticPolicy,
    RecurrentActorCriticCnnPolicy,
    RecurrentMultiInputActorCriticPolicy,
)

MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy

register_policy("MlpPolicy", ActorCriticPolicy)
register_policy("CnnPolicy", ActorCriticCnnPolicy)
register_policy("MultiInputPolicy", MultiInputPolicy)

register_policy("RecurrentMLPPolicy", RecurrentActorCriticPolicy)
register_policy("RecurrentCnnPolicy", RecurrentActorCriticCnnPolicy)
register_policy("RecurrentMultiInputPolicy", RecurrentMultiInputActorCriticPolicy)
