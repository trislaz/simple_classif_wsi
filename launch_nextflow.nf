#!/usr/bin/env nextflow

params.PROJECT_NAME = "tcga_tnbc"
params.PROJECT_VERSION = "pretrained_resnets"
params.y_interest = "LST_status"

// Folders
project_folder  = "./outputs/${params.PROJECT_NAME}_${params.PROJECT_VERSION}"

// Arguments
//repeat = [1, ]
wsi = "/mnt/data4/tlazard/data/tcga_tnbc/images/"
xml = "/mnt/data4/tlazard/data/tcga_tnbc/annotations/annotations_tcga_tnbc_guillaume/"
table_data = "/mnt/data4/tlazard/data/tcga_tnbc/labels_tcga_tnbc.csv"

// Experimental parameters
resolution = Channel.from([0, 1, 2])
n_sample = Channel.from([50, 10, 1])
para = resolution .merge (n_sample)
models = ['resnet18', 'resnet50']
freeze = [0, 1]
pretrained = 1
c_aug = 0 
epochs = 2000
batch_size = 32
patience = 150
repeat = 1..10

process Training {
    publishDir "${output_model_folder}", pattern: "*.pt.tar", overwrite: true
    publishDir "${output_results_folder}", pattern: "*eventsevents.*", overwrite: true
    memory { 30.GB + 5.GB * (task.attempt - 1) }
    errorStrategy 'retry'
    maxRetries 6
    cpus 5
    queue 'gpu-cbio'
    maxForks 5
    clusterOptions "--gres=gpu:1"
    // scratch true
    stageInMode 'copy'

    input:
    tuple val(r), val(sample) from para
    each rep from repeat
    each frozen from freeze
    each model_name from models

    output:
    file("*.pt.tar")

    script:
    python_script = file("./train.py")
    output_model_folder = file("${project_folder}/${model_name}/R_${r}/frozen_$frozen/rep_${rep}/models/")
    tf_folder = file("${project_folder}/${model_name}/R_${r}/frozen_$frozen/rep_${rep}/results/")
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
                          --model_name $model_name \
                          --n_sample $sample \
                          --pretrained $pretrained \
                          --frozen $frozen
    """
}

