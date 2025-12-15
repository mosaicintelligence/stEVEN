# AutoEndo RL: Test, Train, Inference (Headless-Friendly)

## Environment prep (virtual X/GL context needed atm)
- sudo apt-get update && sudo apt-get install -y xvfb

Run with xvfb:
xvfb-run -s "-screen 0 1280x720x24" python stEVEN/autoendo/smoke_test_env.py --use-visualisation --model 0049_H_ABAO_AIOD --insert-vessel celiac_hepatic --steps 50
Training similarly:
xvfb-run -s "-screen 0 1280x720x24" python stEVEN/autoendo/train_sb3.py --algo sac --num-envs 1 --subproc-envs --use-visualisation --device cpu --total-steps 200000

python stEVEN/autoendo/train_sb3.py \
  --algo sac \
  --num-envs 1 \
  --use-visualisation \
  --subproc-envs \
  --device cpu \
  --total-steps 200000


## Quick smoke test (env stability)
- Headless (dummy display):  
  `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python stEVEN/autoendo/smoke_test_env.py --model 0049_H_ABAO_AIOD --insert-vessel celiac_hepatic --steps 50 --use-visualisation`
- Options: `--zero-actions` to send zeros, `--action-scale 0.1` to damp actions, `--use-visualisation` toggles SofaPygame vs dummy vis.

## Training (SB3 PPO/SAC)
- Single env, process isolation, SofaPygame headless:  
  `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python stEVEN/autoendo/train_sb3.py --algo sac --num-envs 1 --subproc-envs --use-visualisation --total-steps 200000 --device cpu`
- Parallel envs (spawned): use `--num-envs 4`.
- Temporal context: `--frame-stack 4` (simple history) or `--recurrent` (PPO LSTM).
- Outputs: checkpoints under `outputs/autoendo/sb3_logs` and final model at `outputs/autoendo/sb3_model.zip` (path configurable via `--save-path`).

## Diffusion policy (behavior cloning)
- Collect + train:  
  `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python stEVEN/autoendo/train_diffusion.py --rollout-steps 25000 --epochs 20 --save-path outputs/autoendo/diffusion_policy.pt --model 0049_H_ABAO_AIOD`
- To imitate an SB3 policy during collection: add `--behavior-path outputs/autoendo/sb3_model.zip --behavior-algo sac`.

## Inference / interactive play (autonomy_train)
- Run with a trained policy (SB3 or diffusion):  
  `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python stEVEN/autoendo/autonomy_train.py --no-vis --policy-path outputs/autoendo/sb3_model.zip --policy-type sb3 --policy-algo sac`
- For diffusion: `--policy-path outputs/autoendo/diffusion_policy.pt --policy-type diffusion`.
- Without a policy: omit `--policy-path` and use keyboard control (visualisation required) or `--no-vis` for random actions.

## Notes
- Headless stability: use `--use-visualisation` with dummy/xvfb; VisualisationDummy can segfault with Sofa on some setups.
- Models/branches: default is random per env reset; fix with `--model` and `--insert-vessel`.
- Action bounds: automatically clipped to the device velocity limits (translation mm/s, rotation rad/s).
