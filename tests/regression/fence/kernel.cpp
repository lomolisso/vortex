#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

static void vecadd(kernel_arg_t* arg) {
	uint32_t count    = arg->task_size;
	int32_t* src0_ptr = (int32_t*)arg->src0_addr;
	int32_t* src1_ptr = (int32_t*)arg->src1_addr;
	int32_t* dst_ptr  = (int32_t*)arg->dst_addr;

	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t offset = gid * count;
	for (uint32_t i = 0; i < count; ++i) {
		dst_ptr[offset+i] = src0_ptr[offset+i] + src1_ptr[offset+i];
	}

	vx_fence();
}

static inline uint32_t cycles() {
	return (uint32_t)csr_read(VX_CSR_MCYCLE);
}

// The add consumes both values, so in-order issue holds the cycle read until
// they have returned.
static inline uint32_t cycles_after(uint32_t a, uint32_t b) {
	uint32_t t;
	asm volatile ("add %0, %1, %2\n\tcsrr %0, %3"
	              : "=&r"(t) : "r"(a), "r"(b), "i"(VX_CSR_MCYCLE));
	return t;
}

static inline uint32_t amo_add(volatile uint32_t* addr, uint32_t value) {
	uint32_t old;
	asm volatile ("amoadd.w %0, %2, (%1)" : "=r"(old) : "r"(addr), "r"(value) : "memory");
	return old;
}

static inline uint32_t amo_read(volatile uint32_t* addr) {
	uint32_t old;
	asm volatile ("amoor.w %0, zero, (%1)" : "=r"(old) : "r"(addr) : "memory");
	return old;
}

static void storm_timer(kernel_arg_t* arg, uint32_t lane, volatile uint32_t* rec) {
	auto ctrl = (volatile uint32_t*)arg->ctrl_addr;
	auto data = (volatile uint32_t*)arg->src0_addr;
	const uint32_t budget  = arg->storm_budget;
	const uint32_t iters   = arg->storm_iters;
	const uint32_t fencers = arg->storm_fencers * vx_num_threads();

	// Time while the storm runs. Greedy warp issue can hold a fencer back until
	// its sibling warps stop fencing, so a start window bounds the wait for all.
	const uint32_t w0 = cycles();
	uint32_t started = amo_read(ctrl + STORM_STARTED);
	while (started < fencers && cycles() - w0 < budget / 8) {
		started = amo_read(ctrl + STORM_STARTED);
	}
	const uint32_t t0 = cycles_after(started, started);

	uint32_t errors = 0, max_lat = 0, last = t0;
	for (uint32_t i = 0; i < iters; ++i) {
		const uint32_t v   = data[i * STORM_ITER_WORDS + lane];
		const uint32_t old = amo_add(ctrl + STORM_COUNTER, 1);
		const uint32_t now = cycles_after(v, old);
		errors += (v != storm_value(i, lane));
		if (now - last > max_lat)
			max_lat = now - last;
		last = now;
	}

	ctrl[STORM_DONE] = 1;

	rec[STORM_REC_COUNT]  = iters;
	rec[STORM_REC_FLAG]   = started;
	rec[STORM_REC_CYCLES] = last - t0;
	rec[STORM_REC_MAXLAT] = max_lat;
	rec[STORM_REC_ERRORS] = errors;
	rec[STORM_REC_WAIT]   = t0 - w0;
	rec[STORM_REC_SEEN]   = 0;
	rec[STORM_REC_LAST]   = 0;
	rec[STORM_REC_KIND]   = STORM_KIND_TIMER;
}

static void storm_fencer(kernel_arg_t* arg, volatile uint32_t* rec) {
	auto ctrl = (volatile uint32_t*)arg->ctrl_addr;
	const uint32_t budget = arg->storm_budget;

	const uint32_t t0 = cycles();
	amo_add(ctrl + STORM_STARTED, 1);

	// Each pass caches the counter's line, so only a fence that invalidates it
	// lets the final read below see the timer's last AMO. The budget bounds the
	// storm when the done flag never arrives.
	uint32_t fences = 0, last, done, elapsed;
	do {
		rec[STORM_REC_COUNT] = fences;
		vx_fence();
		++fences;
		last = ctrl[STORM_COUNTER];
		done = ctrl[STORM_DONE];
		elapsed = cycles() - t0;
	} while (done == 0 && elapsed < budget);

	uint32_t seen = 0;
	if (done != 0) {
		vx_fence();
		seen = ctrl[STORM_COUNTER];
	}

	rec[STORM_REC_COUNT]  = fences;
	rec[STORM_REC_FLAG]   = (done != 0);
	rec[STORM_REC_CYCLES] = elapsed;
	rec[STORM_REC_MAXLAT] = 0;
	rec[STORM_REC_ERRORS] = 0;
	rec[STORM_REC_WAIT]   = 0;
	rec[STORM_REC_SEEN]   = seen;
	rec[STORM_REC_LAST]   = last;
	rec[STORM_REC_KIND]   = STORM_KIND_FENCER;
}

// The first warp of block 0 times its own loads and AMOs while the other
// cores' warps fence in a loop until they see the timer finish. On a device
// with several cores the timer's siblings idle: a fence locks its core's LSU,
// and greedy warp issue can hold the timer back for as long as a sibling keeps
// fencing, which measures the core's issue policy instead of the caches.
static void storm(kernel_arg_t* arg) {
	const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
	auto rec = (volatile uint32_t*)arg->rec_addr + gtid * STORM_REC_WORDS;
	if (gtid < (uint32_t)vx_num_threads()) {
		storm_timer(arg, gtid, rec);
	} else if (blockIdx.x == 0 && gridDim.x > 1) {
		rec[STORM_REC_KIND] = STORM_KIND_IDLE;
	} else {
		storm_fencer(arg, rec);
	}
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
	if (arg->mode == FENCE_MODE_STORM) {
		storm(arg);
	} else {
		vecadd(arg);
	}
}
