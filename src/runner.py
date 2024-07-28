import subprocess 

home = "/HOME"

if __name__ == "__main__":
    pre_training_model= [
            f"{home}/.venvE/bin/python", f"{home}/FINAL_RESULTS/final_code/SimCLR_Scratch_Dev.py"]
    subprocess.run(pre_training_model, check=True)
    methods = ['baseline', 'simclr_inat_adam', 'simclr_animalia_adam','videosimclr_adam']
    percentage_PET = [1,10,100]
    for m in methods: 
        for p in percentage_PET:
            finetune = [
            f"{home}/.venvE/bin/python", f"{home}/FINAL_RESULTS/final_code/finetune_single.py",
            "--percentage_PET", f"{p}",
            "--method",f"{m}"]
            subprocess.run(finetune, check=True)
    plotting = [
        f"{home}/.venvE/bin/python", f"{home}/FINAL_RESULTS/final_code/plotting.py"]
    subprocess.run(plotting, check=True)

    

    