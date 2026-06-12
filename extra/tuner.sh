#!/bin/bash

FILE="extra/minimal.cpp"

if [ ! -f "$FILE" ]; then
    echo "Error: $FILE not found!"
    exit 1
fi

echo "================================================="
echo " Starting Auto-Tuning for myGEMM4 on AMD RDNA3   "
echo "================================================="
echo -e "TS\tWPT\tGFLOPS (Average)\tSpread"
echo "-------------------------------------------------"

for TS_VAL in 16 32; do
    for WPT_VAL in 4 8; do
        
        sed -i "s/^#define TS .*/#define TS $TS_VAL/" $FILE
        sed -i "s/^#define WPT .*/#define WPT $WPT_VAL/" $FILE
        
        g++ -O3 -Wall -std=c++11 $FILE -o bin/minimal -lOpenCL
        
        if [ $? -ne 0 ]; then
            echo -e "$TS_VAL\t$WPT_VAL\t[COMPILATION FAILED]"
            continue
        fi

        OUTPUT=$(./bin/minimal)
        
        if echo "$OUTPUT" | grep -q "ERROR"; then
            echo -e "$TS_VAL\t$WPT_VAL\t[OPENCL BUILD FAILED]"
            continue
        fi

        PERF=$(echo "$OUTPUT" | grep "Performance:" | awk '{print $3}')
        SPREAD=$(echo "$OUTPUT" | grep "Performance:" | awk '{print $5}' | tr -d '()')
        
        if [ -z "$PERF" ]; then
             echo -e "$TS_VAL\t$WPT_VAL\t[RUNTIME FAILED]"
        else
             echo -e "$TS_VAL\t$WPT_VAL\t$PERF\t\t$SPREAD"
        fi
        
    done
done

echo "================================================="
echo " Tuning completed. Reverting to safe defaults.   "
echo "================================================="
sed -i "s/^#define TS .*/#define TS 16/" $FILE
sed -i "s/^#define WPT .*/#define WPT 8/" $FILE
