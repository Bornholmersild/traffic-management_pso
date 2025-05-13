from pso_traffic_optimizer import PSO_TrafficOptimizer

if __name__ == "__main__":    
    #network = ["sumo/world_line_1_tls.net.xml", "sumo/vehicles_line_1_tls.rou.xml"]
    #network = ["sumo/world_line_10_tls.net.xml", "sumo/vehicles_line_10_tls.rou.xml"]

    #network = ["sumo/world_cross_1_tls_2_phases.net.xml", "sumo/cross_world_debug_tester.rou.xml"]
    #network = ["sumo/world_cross_1_tls_2_phases_2_lanes.net.xml", "sumo/vehicles_cross_1_tls_2_phases_2_lanes.rou.xml"]

    #network = ["sumo/world_grid_5_tls_2_lanes.net.xml", "sumo/vehicles_grid_5_tls_2_lanes.rou.xml"]
    #network = ["sumo/world_grid_5_tls_2_lanes.net.xml", "sumo/vehicles_grid_5_tls_2_lanes_heavyload.rou.xml"]

    #network = ["sumo/real_world_odense.sumocfg", "sumo/vehicles_odense.rou.xml"]
    network = ["sumo/world_odense.net.xml", "sumo/vehicles_odense.rou.xml"]
    base_path_to_save = "output"

    
    #-------------------------------------------------------------------------------------#
    # NOTE: 
    # The seeds are used to generate the random number generator and 
    # need to be chanced if not all indenpendent runs are executed at once!!!
    
    select_st_seed = 1              # Select which seed to start with
    num_independent_runs = 1        # Select number of independent runs to execute  
    #-------------------------------------------------------------------------------------#
    
    start_run = select_st_seed - 1                  
    end_run = start_run + num_independent_runs
    seeds = [1, 2, 3, 4, 5]  

    for run in range(start_run, end_run):
        seed = seeds[run]
        print(f"Run nr: {run} and seed value: {seed}")
        pso = PSO_TrafficOptimizer(
                                network,
                                random_seed=seed,
                                sim_iterations=1500,     # 1500
                                num_particles=10,        # 10
                                iterations_max=100,       # 100
                                w_max=0.5,
                                w_min=0.1,
                                c1=2,
                                c2=2,
                                phase_min=10,
                                phase_max=40,
                                lamda_factor=0.5,
                                gui_on=False
                                )
        
        pso.run(run, base_path_to_save)
        

        
        