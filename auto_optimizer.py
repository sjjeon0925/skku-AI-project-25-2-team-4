import subprocess
import itertools
import sys
import argparse

# final parameters
PARAM_GRID = {
    'baseline': {
        'mlp_epochs': [300]
    },
    'gnn_only': {
        'gnn_dim': [16],
        'gnn_epochs': [500],
        'gnn_lr': [0.005]
    },
    'proposed': {
        'gnn_dim': [16],
        'gnn_epochs': [500],
        'mlp_epochs': [300]
    }
}

def run_command(cmd):
    """쉘 명령어를 실행하고 출력을 반환. 에러 시 None 반환"""
    print(f"cmd> {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"Error: {stderr}")
        return None # 에러 발생 시 None 반환
    return stdout

def parse_rmse(output):
    """train.py CV 결과에서 RMSE 파싱"""
    if output is None: return 999.0
    for line in output.split('\n'):
        if "CV Result" in line:
            return float(line.split("Average RMSE:")[1].strip())
    return 999.0

def run_process(task_mode):
    """
    task_mode:
      - 'opt': Grid Search를 통해 최적 파라미터 탐색 후 학습/평가
      - 'run': PARAM_GRID의 첫 번째 값을 최적값으로 간주하고 즉시 학습/평가
    """
    print(f"\n🚀 Starting Process in [{task_mode.upper()}] mode...")

    for mode, grid in PARAM_GRID.items():
        print(f"\n========== Target Model: [{mode}] ==========")
        
        best_score = 999.0
        best_params = {}

        # -------------------------------------------------------
        # MODE 1: Optimization (Grid Search + CV)
        # -------------------------------------------------------
        if task_mode == 'opt':
            print(f"🔍 [Opt] Searching for best parameters...")
            keys, values = zip(*grid.items())
            combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
            
            for params in combinations:
                cmd = f"python3 train.py --mode {mode} --job cv --k_fold 3"
                for k, v in params.items():
                    cmd += f" --{k} {v}"
                
                output = run_command(cmd)
                
                if output is None:
                    print(f"   Params: {params} -> Failed ❌")
                    continue
                    
                rmse = parse_rmse(output)
                print(f"   Params: {params} -> RMSE: {rmse:.4f}")
                
                if rmse < best_score:
                    best_score = rmse
                    best_params = params
            
            if not best_params and combinations:
                print(f"⚠️ {mode} 모드의 모든 CV 실행이 실패했습니다.")
                continue
                
            print(f"🏆 Best Params Found: {best_params} (RMSE: {best_score:.4f})")

        # -------------------------------------------------------
        # MODE 2: Direct Run (Skip CV)
        # -------------------------------------------------------
        else: # task_mode == 'run'
            # 리스트의 첫 번째 값만 가져와서 설정
            best_params = {k: v[0] for k, v in grid.items()}
            print(f"⏩ [Run] Skipping optimization. Using config: {best_params}")

        # -------------------------------------------------------
        # Common Step: Final Training & Evaluation
        # -------------------------------------------------------
        print(f"🔥 [Train] Starting final training with best params...")
        model_name = f"best_{mode}"
        
        train_cmd = f"python3 train.py --mode {mode} --job train --model_name {model_name}"
        for k, v in best_params.items():
            train_cmd += f" --{k} {v}"
            
        train_out = run_command(train_cmd)
        if train_out is None:
            print(f"❌ Final Training Failed for {mode}")
            continue 
        
        print(f"📊 [Eval] Starting evaluation...")
        eval_cmd = f"python3 evaluate.py --mode {mode} --model_name {model_name}"
        eval_out = run_command(eval_cmd)
        
        if eval_out:
            print(f"-------- Evaluation Result ({mode}) --------")
            print(eval_out)
            print("--------------------------------------------")
        else:
            print(f"❌ Evaluation Failed for {mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 'opt' 또는 'run' 중 하나를 선택하도록 설정 (기본값: run)
    parser.add_argument('--task', type=str, choices=['opt', 'run'], default='run', 
                        help="opt: Grid Search 수행 / run: 설정된 값으로 즉시 학습")
    
    args = parser.parse_args()
    run_process(args.task)