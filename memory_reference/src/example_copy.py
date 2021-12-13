from context_manager import sync_open

def example_manual_copy(algos, size):
     for algo in algos:   
        algo.whoami()
        algo.initialize_array(size)

        input = algo.get_input_memory()
        output = algo.get_output_memory()

        if(algo.is_synced_mem() == True):
            input.read()
            input_temp = input.buffer
            output_temp = output.buffer
        else :
            input_temp = input
            output_temp = output

        print(input_temp)
        print(type(input_temp))

        for i in range(size):
            input_temp[i] = 2.0

        if(algo.is_synced_mem() == True): 
            input.write()

        algo.compute(input, output)

        if(algo.is_synced_mem() == True): 
            output.read()

        print(output_temp)
        print(type(output_temp))

def example_context_copy(algos, size):
    for algo in algos:
        
        algo.whoami()
        algo.initialize_array(size)

        with sync_open(algo.get_input_memory(), algo.is_synced_mem()) as input:
            for i in range(size):
                input[i] = 2.0

        algo.compute()
                
        with sync_open(algo.get_output_memory(), algo.is_synced_mem()) as output:
            print(output)