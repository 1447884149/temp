set -e
cd ../../../ #到federatedscope目录
# basic configuration
# cd /data/yhp2022/FS/federatedscope/model_heterogeneity/SFL_methods/FedProto_rethink_cluster
gpu=6
result_folder_name=FedProto_rethink_cluster2
global_eval=False
local_eval_whole_test_dataset=True
method=FedProto_rethink_cluster
script_floder="model_heterogeneity/SFL_methods/"${method}
result_floder=model_heterogeneity/result/${result_folder_name}
# common hyperparameters
dataset='citeseer'
total_client=(3 5 10)
local_update_step=(4 16 32)
optimizer='SGD'
seed=(0 1 2)
lrs=(0.05 0.1 0.25)
total_round=200
patience=60
momentum=0.9
freq=1
pass_round=0
# Local-specific parameters
proto_weight=(0.1)
lamda=(0.1 0.5 1.0)
# Define function for model training
cnt=0
train_model() {
  python main.py --cfg ${main_cfg} \--client_cfg ${client_cfg} \
    federate.client_num ${1} \
    federate.make_global_eval ${global_eval} \
    data.local_eval_whole_test_dataset ${local_eval_whole_test_dataset} \
    seed ${2} \
    train.local_update_steps ${3} \
    train.optimizer.lr ${4} \
    federate.total_round_num ${total_round} \
    train.optimizer.type ${optimizer} \
    train.optimizer.momentum ${momentum} \
    device ${gpu} \
    early_stop.patience ${patience} \
    result_floder ${result_floder} \
    exp_name ${exp_name} \
    eval.freq ${freq} \
    fedproto.proto_weight ${5} \
    fedproto.lamda ${6}
}

# Loop over parameters for HPO
for data in "${dataset[@]}"; do
  for client_num in "${total_client[@]}"; do
    for lr in "${lrs[@]}"; do
      for ls in "${local_update_step[@]}"; do
        for weight in "${proto_weight[@]}"; do
          for s in "${seed[@]}"; do
            for lamda in "${lamda[@]}"; do
              let cnt+=1
              if [ "$cnt" -lt $pass_round ]; then
                continue
              fi
              main_cfg=$script_floder"/"$method"_on_"$data".yaml"
              client_cfg="model_heterogeneity/model_settings/"$client_num"_Heterogeneous_GNNs.yaml"
              exp_name="SFL_HPO_"$method"_on_"$data"_"$client_num"_clients"
              train_model "$client_num" "$s" "$ls" "$lr" "$weight" "$lamda"
            done
          done
        done
      done
    done
  done
done
