
STORAGE="40000 10000 8000 6000 4000 2000 500"
METHOD="sim var"

TZ="GMT-8" date +"%Y-%m-%d %H:%M:%S"

for c in $STORAGE
do
   python3 rp_rd.py -c "$c"
   python3 rp_fifo.py -c "$c"

   for m in $METHOD
   do
       python3 rp_gd.py -c "$c" -m "$m"
       python3 rp_fptas.py -c "$c" -m "$m"
   done
   TZ="GMT-8" date +"%Y-%m-%d %H:%M:%S"
done

echo "Done"


for c in $STORAGE
do
    python3 qa_merged.py -c "$c" -m rd
    python3 qa_merged.py -c "$c" -m fifo

    for m in $METHOD
    do
        python3 qa_merged.py -c "$c" -m "$m" -a gd
        python3 qa_merged.py -c "$c" -m "$m" -a fptas
    done
    TZ="GMT-8" date +"%Y-%m-%d %H:%M:%S"
done

echo "Done"