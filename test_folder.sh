base_folder="loss_conf_sweep";

input=$(<test_inputs.txt);
for dir in training_dir/"$base_folder"/*/; do
  mkdir "$dir"preds;
  python demo/demo.py --config-file "$dir"config.yaml --input $input --output "$dir"preds/ --opts MODEL.WEIGHTS "$dir"/model_final.pth TEST.TEST_ALL_CHECKPOINTS True
done
