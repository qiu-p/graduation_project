import sys
sys.path.append("..")
from o0_logger import logger
import o1_environment_adder as environment
import o2_policy_adder as policy 
from o2_policy_adder import BasicBlock
import o3_trainer_adder as trainer
from o3_trainer_adder import ReplayMemory
from o5_utils import setup_logger, set_global_seed
import curses
import hydra
from omegaconf import DictConfig, OmegaConf

"""
    # env kwargs
        bit_width
    # policy kwargs
        bit_width
        device
    # trainer kwargs 
        device
    # exp kwargs
        exp_prefix
        base_log_dir
"""

DEVICE = 'cuda:0'

@hydra.main(config_path="configs_adder")
def main(cfg: DictConfig) -> None:
    seed = set_global_seed(cfg.config_groups.logger_exp_kwargs.seed)

    # 1. init q policy 
    q_policy = getattr(policy, cfg.config_groups.policy['class'])(
                BasicBlock,
                device=DEVICE,
                **cfg.config_groups.policy.kwargs).to(DEVICE)
    target_q_policy = getattr(policy, cfg.config_groups.policy['class'])(
                BasicBlock,
                device=DEVICE,
                **cfg.config_groups.policy.kwargs).to(DEVICE)

    # 2. init environment
    env = getattr(environment, cfg.config_groups.environment['class'])(
        seed,
        q_policy,
        **cfg.config_groups.environment.kwargs)
    # 3. init replay memory
    replay_buffer = ReplayMemory()
    # 4. init trainer
    train_agent = getattr(trainer, cfg.config_groups.trainer['class'])(
        env,
        q_policy,
        target_q_policy,
        replay_buffer,
        device=DEVICE,
        **cfg.config_groups.trainer.kwargs
    )

    # 5. setup logger
    logger.reset()
    cfg_dict = OmegaConf.to_container(cfg)
    variant = cfg_dict
    cfg.config_groups.logger_exp_kwargs.seed = seed
    actual_log_dir = setup_logger(
        variant=variant,
        **cfg.config_groups.logger_exp_kwargs
    )

    # 6. run experiments
    train_agent.run_experiments()

if __name__ == "__main__":
    main()
