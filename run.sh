#!/bin/bash

Num_LOOPS=${NUM_LOOPS:-1}

for ((i=1; i<=NUM_LOOPS; i++)); do
    echo "Starting to run $NUM_LOOPS simulations."

    python3 main.py \
        --N ${N:-100} \
        --num_conspirators_frac ${NUM_CONSPIRATORS_FRAC:-0.05} \
        --conspirator_bool ${CONSPIRATOR_BOOL:-False} \
        --true_mega_node_bool ${TRUE_MEGA_NODE_BOOL:-False} \
        --consp_mega_node_bool ${CONSP_MEGA_NODE_BOOL:-False} \
        --num_iterations ${NUM_ITERATIONS:-150} \
        --num_simulations ${NUM_SIMULATIONS:-200} \
        --k ${K:-$(echo "0.1 * ${N:-100}" | bc)} \
        --m ${M:-5} \
        --graph ${GRAPH:-"ER"} \
        --cap ${CAP:-1.0} \
        --sigmoid_factor ${SIGMOID_FACTOR:-4.0} \
        --std_draw ${STD_DRAW:-0.5} \
        --std_likelihood ${STD_LIKELIHOOD:-0.5} \
        --flex_strength ${FLEX_STRENGTH:-0.5} \
        --log_belief_bool ${LOG_BELIEF_BOOL:-False} \
        --confbias_bool ${CONFBIAS_BOOL:-True}
    LEFT=$((NUM_LOOPS - i))
    echo "Loop number $i is done, $LEFT"
done