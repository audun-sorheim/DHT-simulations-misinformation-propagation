#!/bin/bash

NUM_LOOPS=${NUM_LOOPS:-10}

for ((i=1; i<=NUM_LOOPS; i++)); do
    echo "Starting to run $NUM_LOOPS simulations."
    K=${K:-$(awk "BEGIN {print 0.1 * ${N:-100}}")}
    FLEX_INTERVAL=${FLEX_INTERVAL:-"0.0 0.0"}

    python3 main.py \
        --N ${N:-4096} \
        --num_conspirators_frac ${NUM_CONSPIRATORS_FRAC:-0.05} \
        $( [[ "${CONSPIRATOR_BOOL:-False}" == "True" ]] && echo "--conspirator_bool" ) \
        $( [[ "${TRUE_MEGA_NODE_BOOL:-False}" == "True" ]] && echo "--true_mega_node_bool" ) \
        $( [[ "${CONSP_MEGA_NODE_BOOL:-False}" == "True" ]] && echo "--consp_mega_node_bool" ) \
        --num_iterations ${NUM_ITERATIONS:-200} \
        --num_simulations ${NUM_SIMULATIONS:-20} \
        --k ${K:-10} \
        --m ${M:-5} \
        --graph ${GRAPH:-"BA"} \
        --cap ${CAP:-1.0} \
        --sigmoid_factor ${SIGMOID_FACTOR:-4.0} \
        --s ${S:-0.6} \
        --std_draw ${STD_DRAW:-0.75} \
        --std_likelihood ${STD_LIKELIHOOD:-0.75} \
        --flex_strength ${FLEX_STRENGTH:-0.8} \
        --flex_interval ${FLEX_INTERVAL} \
        $( [[ "${LOG_BELIEFS_BOOL:-False}" == "True" ]] && echo "--log_beliefs_bool" ) \
        $( [[ "${CONFBIAS_BOOL:-True}" == "True" ]] && echo "--confbias_bool" ) \
        $( [[ "${GAUSSIAN_BOOL:-True}" == "True" ]] && echo "--gaussian_bool" ) \
        $( [[ "${SAVE_ALL:-False}" == "True" ]] && echo "--save_all" ) \
        --dir ${DIR:-"test"} \

    LEFT=$((NUM_LOOPS - i))
    echo "Loop number $i is done, $LEFT"
done
