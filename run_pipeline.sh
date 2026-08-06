cd /opt/domain-adaptation-people-counting/src

for script in \
    main_regression_pretrain.py \
    parameter_based_methods.py \
    feature_based_methods.py \
    instance_based_methods.py \
    all_transfer_combinations.py
do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶  Running: $script"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python "$script" && echo "✓  $script done" || echo "✗  $script FAILED"
done
