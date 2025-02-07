#include "profiler.h"

Profiler profiler = {0};
static clock_t current_start;

void init_profiler(void) {
    profiler.num_stats = 0;
    profiler.enabled = false;
}

void enable_profiler(bool enable) {
    profiler.enabled = enable;
}

void start_profile(const char* name) {
    if (!profiler.enabled) return;
    current_start = clock();
}

void end_profile(const char* name, double flops) {
    if (!profiler.enabled) return;
    
    clock_t end = clock();
    double time = ((double)(end - current_start)) / CLOCKS_PER_SEC;
    
    int idx = -1;
    for (int i = 0; i < profiler.num_stats; i++) {
        if (strcmp(profiler.stats[i].name, name) == 0) {
            idx = i;
            break;
        }
    }
    
    if (idx == -1) {
        idx = profiler.num_stats++;
        profiler.stats[idx].name = name;
        profiler.stats[idx].total_time = 0;
        profiler.stats[idx].total_flops = 0;
        profiler.stats[idx].calls = 0;
    }
    
    profiler.stats[idx].total_time += time;
    profiler.stats[idx].total_flops += flops;
    profiler.stats[idx].calls++;
}

void print_profile_stats(void) {
    if (!profiler.enabled) return;
    
    printf("\nProfiling Results:\n");
    printf("%-25s %12s %12s %10s %15s\n", 
           "Component", "Time (ms)", "GFLOPS", "Calls", "GFLOPS/sec");
    printf("----------------------------------------------------------------\n");
    
    double total_time = 0;
    double total_flops = 0;
    
    for (int i = 0; i < profiler.num_stats; i++) {
        ProfileStat* stat = &profiler.stats[i];
        double gflops = stat->total_flops / 1e9;
        double gflops_per_sec = gflops / stat->total_time;
        
        printf("%-25s %11.2f %11.2f %10d %15.2f\n",
               stat->name,
               stat->total_time * 1000,
               gflops,
               stat->calls,
               gflops_per_sec);
               
        total_time += stat->total_time;
        total_flops += stat->total_flops;
    }
    
    printf("----------------------------------------------------------------\n");
    printf("%-25s %11.2f %11.2f %10s %15.2f\n",
           "Total",
           total_time * 1000,
           total_flops / 1e9,
           "-",
           (total_flops / 1e9) / total_time);
} 