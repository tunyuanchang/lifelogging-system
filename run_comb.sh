
STORAGE="40000 10000 2000 500" #MB

TZ="GMT-8" date +"%Y-%m-%d %H:%M:%S"

for c in $STORAGE
do
    python3 combined_rd.py -c "$c" -m rd -n True
    python3 combined_fifo.py -c "$c" -m fifo -n True
    python3 combined_gd.py -c "$c" -m sim -a gd -n True
    TZ="GMT-8" date +"%Y-%m-%d %H:%M:%S"
done

echo "Done"