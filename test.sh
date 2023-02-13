#!/bin/bash

A="I'm looking for contributors who can do optimizations better than me."
read -r -d '' B << EOM
Then took the other, as just as fair,
And having perhaps the better claim,
Because it was grassy and wanted wear;
Though as for that the passing there
Had worn them really about the same,
EOM

mkdir optimized_examples/A -p
mkdir optimized_examples/B -p

save() {
	# check $1 not nul
	if [ -z "$1" ]; then
		echo "save: \$1 is empty"
		exit 1
	fi
	[ -d "optimized_examples/A/$1" ] && rm -rf "optimized_examples/A/$1"
	[ -d "optimized_examples/B/$1" ] && rm -rf "optimized_examples/B/$1"
	mv A optimized_examples/A/$1
	mv B optimized_examples/B/$1
}
f() {
	python tortoise/do_tts.py --text "$A" --voice emma --seed 42 "$@" | grep 'took .* seconds' -o | grep '[0-9]*\.[0-9]*' -o | tee results.txt
	mv results.txt results
	mv results A
	sleep 5; # sleep to ensure gpu cools down
	python tortoise/do_tts.py --text "$B" --voice emma --seed 42 "$@" | grep 'took .* seconds' -o | grep '[0-9]*\.[0-9]*' -o | tee results.txt
	mv results.txt results
	mv results B
	sleep 10; # sleep to ensure gpu cools down
}


f --preset ultra_fast_old --seed 42 --no_cache --low_vram
save tortoise_original-with_original_vram
f --preset ultra_fast_old --seed 42 --no_cache
save tortoise_original
f --preset ultra_fast_old --seed 42 --no_cache --half # autocast to fp16 on autoregression + CLVP
save tortoise_original-half_incomplete
f --preset ultra_fast_old --seed 42 --kv_cache # kv_cache
save tortoise_original-kv_cache

f --preset ultra_fast --seed 42 --no_cache # aka DPM++2M with 10 steps and cond_free
save ultra_fast
f --preset ultra_fast --seed 42 --no_cache --half # autocast + DPM++2M
save ultra_fast-half
f --preset ultra_fast --seed 42 --no_cache --no_cond_free # aka DPM++2M without cond_free
save ultra_fast-no_cond_tree
f --preset ultra_fast --seed 42 --kv_cache # DPM++2M with kv_cache
save ultra_fast-kv_cache
f --preset ultra_fast --seed 42 --kv_cache --half # DPM++2M with kv_cache + autocast
save ultra_fast-kv_cache-half
f --preset ultra_fast --seed 42 --no_cache --half --no_cond_free # DPM++2M (no cond_free) + autocast
save ultra_fast-half-no_cond_tree
f --preset ultra_fast --seed 42 --kv_cache --half --no_cond_free # enable all optimizations
save ultra_fast-kv_cache-half-no_cond_tree

# f --text ... # some future command that invokes full autocast
