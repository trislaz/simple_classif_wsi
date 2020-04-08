#!/usr/bin/env nextflow

params.PROJECT_NAME = "tcga_tnbc"
params.PROJECT_VERSION = "simple_classif_repeat_experiment"
params.resolution = "2"
params.y_interest = "LST_status"
model = "resnet18"

// Folders
project_folder  = "./outputs/${params.PROJECT_NAME}_${params.PROJECT_VERSION}"

// Arguments
//repeat = [1, ]
wsi = "/mnt/data4/tlazard/data/tcga_tnbc/images/"
xml = "/mnt/data4/tlazard/data/tcga_tnbc/annotations/annotations_tcga_tnbc_guillaume/"
table_data = "/mnt/data4/tlazard/data/tcga_tnbc/labels_tcga_tnbc.csv"
resolution = [1, 2]
n_sample = [1]
color_aug = [0]
epochs = 2000
batch_size = 32
patience = 150

process Training {
    publishDir "${output_model_folder}", pattern: "*.pt.tar", overwrite: true
    publishDir "${output_results_folder}", pattern: "*eventsevents.*", overwrite: true
    memory { 30.GB + 5.GB * (task.attempt - 1) }
    errorStrategy 'retry'
    maxRetries 6
    cpus 5
    queue 'gpu-cbio'
    maxForks 6
    clusterOptions "--gres=gpu:1"
    // scratch true
    stageInMode 'copy'

    input:
    val rep from 1..25 
    each r from resolution
    each sample from n_sample
    each c_aug from color_aug

    output:
    file("*.pt.tar")

    script:
    python_script = file("./train.py")
    output_model_folder = file("${project_folder}/${model}_R_${r}_nsample_${sample}_caug_${c_aug}/rep_${rep}/models/")
    tf_folder = file("${project_folder}/${model}_R_${r}_nsample_${sample}_caug_${c_aug}/rep_${rep}/results/")
    tf_folder.mkdir()

    """
    export EVENTS_TF_FOLDER=${tf_folder}
    module load cuda10.0
    python $python_script --wsi $wsi \
                          --xml $xml \
                          --table_data $table_data \
                          --batch_size $batch_size \
                          --epochs $epochs \
                          --patience $patience \
                          --color_aug $c_aug \
                          --resolution $r \
                          --model $model \
                          --n_sample $sample \
    """
}

