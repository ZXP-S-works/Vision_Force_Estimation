run()
{
script=run.sh

args="
--train_data_dir=data/${train_set}
--valid_data_dir=data/${test_set}
--train_set_size=${train_set_size}
--n_input_channel=${n_input_channel}
--n_output_channel=${n_output_channel}
--n_hidden=${n_hidden}
--env_type=${env_type}
--device=cuda
--kernel_size=3
--note=${note}
--resolution=${resolution}
--network=resnet
--rot_aug=${rot_aug}
--trans_aug=${trans_aug}
"

./run.sh ${args}
}

for runs in 1 2
  do
    for resolution in 128 512
    do
        n_hidden=32
        env_type=simulation_force
        test_set='simulation_512_1000.pt'
        train_set='simulation_512_5000.pt'
        n_input_channel=3
        n_output_channel=2
        train_set_size=5000
        rot_aug=0
        trans_aug=0
        note=${env_type}_${resolution}
        run
    done
  done