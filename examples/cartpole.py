from easyfl.coordinator_rl import init_rl, run_rl

config = {
    "model": "dqn"
}
init_rl(config)
run_rl()
