localparam W = 32; // Width of inputs of tasks, make bigger if for example an address width exceeds 32 bits

// vector matrix multiplication
task instr_vecMatMult
    (
        input [W-1:0] memIN1,
        input [W-1:0] memIN2,
        input [W-1:0] memOUT,
        
        input [W-1:0] addr_st_IN1,
        input [W-1:0] addr_end_IN1,
        
        input [W-1:0] addr_st_IN2,
        input [W-1:0] addr_end_IN2,
        
        input [W-1:0] addr_st_OUT,
        input [W-1:0] addr_end_OUT
    );
    
    begin
    instr_type <= IT_MATH_OP;
    
    op_select <= VECMATMULT;
    
    mem_select[IN1] <= memIN1; // in1
    mem_select[IN2] <= memIN2; // in2
    mem_select[OUT] <= memOUT; // out
    
    addr_st[IN1] <= addr_st_IN1;
    addr_end[IN1] <= addr_end_IN1;
    
    addr_st[IN2] <= addr_st_IN2;
    addr_end[IN2] <= addr_end_IN2;
    
    addr_st[OUT] <= addr_st_OUT;
    addr_end[OUT] <= addr_end_OUT;
    
    vmm_addr_vecin_st_offset <= 0;
    end
endtask

// elementwise vector-vector addition/multiplication
task instr_vecVecFunc
    (
        input [W-1:0] func, // MULT ADD
        
        input [W-1:0] memIN1,
        input [W-1:0] memIN2,
        input [W-1:0] memOUT,
        
        input [W-1:0] addr_st_IN1,
        input [W-1:0] addr_st_IN2,
        input [W-1:0] addr_st_OUT,
        
        input [W-1:0] num_pos,
        
        input disableAddrVecout // Generally zero, set to 1 when the inputs and output are the same memory
    );
    
    begin
    instr_type <= IT_MATH_OP;

    mem_select[IN1] <= memIN1; // in1
    mem_select[IN2] <= memIN2; // in2
    mem_select[OUT] <= memOUT; // out

    op_select <= VECVECFUNC;
    
    vecVecFunc_select <= func; 
    
    addr_st[IN1] <= addr_st_IN1;
    addr_st[IN2] <= addr_st_IN2;
    addr_st[OUT] <= addr_st_OUT;
    
    num_positions <= num_pos;
    
    vvf_disable_addr_vecout <= disableAddrVecout;
    end
endtask

// elementwise vector sigm/tanh/1-a
task instr_vecFunc
    (
        input [W-1:0] func, // SIGM TANH ONEMA
        
        input [W-1:0] memIN,
        input [W-1:0] memOUT,
        
        input [W-1:0] addr_st_IN,
        input [W-1:0] addr_st_OUT,
        
        input [W-1:0] num_pos,
        
        input [W-1:0] addr_in_start_offset // Generally zero, used to reorder data after a vecVecFunc when the inputs and outputs are from the same memory
    );
    
    begin
    instr_type = IT_MATH_OP;
    
    mem_select[IN1] = memIN; // in1
    mem_select[OUT] = memOUT; // out

    op_select = VECFUNC;
    vecFunc_select = func;
    
    addr_st[IN1] = addr_st_IN;
    addr_st[OUT] = addr_st_OUT;
    
    num_positions = num_pos;
    
    vf_addr_vecin_st_offset = addr_in_start_offset;
    end
endtask


/*
Calls:

instr_vecMatMult(in1, in2, out, // in1, in2, out
                 stIn1, endIn1, // address start/end in1 
                 stIn2, endIn2, // address start/end in2 
                 stOut, endOut); // address start/end out

instr_vecVecFunc(MULT, // MULT ADD
                 in1, in2, out, // in1, in2, out
                 stIn1, stIn2, stOut, // address start in1, in2, out
                 numPositions, // number of positions
                 0); // Set to 1 when in1, in2 and out are ports from the same memory

instr_vecFunc(SIGM, // SIGM TANH ONEMA
              in, out, // in, out
              stIn, stOut, // address start in, out
              numPositions, // number of positions
              0); // Set to 0 if data is ordered (which is most of the time)
*/

always @(posedge clk)
if (en_prog)
case (prog_counter)
    // GRU layer
    /// First cycle of GRU layer

    8'd0: begin // [xz] = vecMatMult(inData[0], gruKernel[0 . .])
        instr_vecMatMult(MEMPARAMS, MEMIN, MEMTEMPA, // in1, in2, out
                         addr_st_gk, addr_st_gk + memdepth_gk_individual, // address start/end in1 
                         0, feature_size, // address start/end in2 
                         0, gru_out_size); // address start/end out
        
        addr_indata_offset <= feature_size; // For the next cycles
    end
    
    8'd1: begin // [xh] = vecMatMult(inData[0], gruKernel[. . 0])
        instr_vecMatMult(MEMPARAMS, MEMIN, MEMTEMPA, // in1, in2, out
                         addr_st_gk + 2*memdepth_gk_individual, addr_st_gk + 3*memdepth_gk_individual, // address start/end in1 
                         0, feature_size, // address start/end in2 
                         gru_out_size, 2*gru_out_size); // address start/end out
    end

    8'd2: begin // [xz] = vecVecAdd([xz], gruBias[0 . .])
        instr_vecVecFunc(ADD, // MULT ADD
                         MEMPARAMS, MEMTEMPB, MEMTEMPA, // in1, in2, out
                         addr_st_gb, 0, 0, // address start in1, in2, out
                         gru_out_size, // number of positions
                         0); // Set to 1 when in1, in2 and out are ports from the same memory
    end
    
    8'd3: begin // [xh] = vecVecAdd([xh], gruBias[. . 0])
        instr_vecVecFunc(ADD, // MULT ADD
                         MEMPARAMS, MEMTEMPB, MEMTEMPA, // in1, in2, out
                         addr_st_gb + 2*gru_out_size, gru_out_size, gru_out_size, // address start in1, in2, out
                         gru_out_size, // number of positions
                         0); // Set to 1 when in1, in2 and out are ports from the same memory
    end
    
    8'd4: begin // [xz] = sigm([xz])
        instr_vecFunc(SIGM, // SIGM TANH ONEMA
                      MEMTEMPB, MEMTEMPA, // in, out
                      0, 0, // address start in, out
                      gru_out_size, // number of positions
                      0); // Set to 0 if data is ordered (which is most of the time)
    end
    
    8'd5: begin // [xh] = tanh([xh])
        instr_vecFunc(TANH, // SIGM TANH ONEMA
                      MEMTEMPB, MEMTEMPA, // in, out
                      gru_out_size, gru_out_size, // address start in, out
                      gru_out_size, // number of positions
                      0); // Set to 0 if data is ordered (which is most of the time)
    end
    
    8'd6: begin // [xz] = 1  - [xz]
        instr_vecFunc(ONEMA, // SIGM TANH ONEMA
                      MEMTEMPB, MEMTEMPA, // in, out
                      0, 0, // address start in, out
                      gru_out_size, // number of positions
                      0); // Set to 0 if data is ordered (which is most of the time)
    end
    
    8'd7: begin // [htm1] = vecVecMult([xz], [xh])
        instr_vecVecFunc(MULT, // MULT ADD
                         MEMTEMPA, MEMTEMPB, MEMTEMPC, // in1, in2, out
                         0, gru_out_size, 0, // address start in1, in2, out
                         gru_out_size, // number of positions
                         0); // Set to 1 when in1, in2 and out are ports from the same memory
    end
    
    /// Other cycles of GRU layer
    
    8'd8: begin // [xz, xr, xh] = vecMatMult(inData[i], gruKernel)
        instr_vecMatMult(MEMPARAMS, MEMIN, MEMTEMPA, // in1, in2, out
                         addr_st_gk, addr_st_gk + memdepth_gk, // address start/end in1 
                         0 + addr_indata_offset, feature_size + addr_indata_offset, // address start/end in2 
                         0, 3*gru_out_size); // address start/end out
        
        addr_indata_offset <= addr_indata_offset + feature_size; // For the next cycle
    end
    
    8'd9: begin // [xz, xr, xh] = vecVecAdd([xz, xr, xh], gruBias)
        instr_vecVecFunc(ADD, // MULT ADD
                         MEMPARAMS, MEMTEMPB, MEMTEMPA, // in1, in2, out
                         addr_st_gb, 0, 0, // address start in1, in2, out
                         memdepth_gb, // number of positions
                         0); // Set to 1 when in1, in2 and out are ports from the same memory
    end
    
    8'd10: begin // [recz, recr] = vecMatMult(htm1, gruRecKernel[0 0 .])
        instr_vecMatMult(MEMPARAMS, MEMTEMPC, MEMTEMPA, // in1, in2, out
                         addr_st_grk, addr_st_grk + 2*memdepth_grk_individual, // address start/end in1 
                         0, gru_out_size, // address start/end in2 
                         3*gru_out_size, 3*gru_out_size + 2*gru_out_size); // address start/end out
    end
    
    8'd11: begin // [recz, recr] = vecVecAdd([xz, xr], [recz, recr]) // All output data address is offset by a value which is equal to the latency of vecVecFunc (it wraps around)
        instr_vecVecFunc(ADD, // MULT ADD
                         MEMTEMPB, MEMTEMPA, MEMTEMPA, // in1, in2, out
                         0, 3*gru_out_size, 0, // address start in1, in2, out
                         2*gru_out_size, // number of positions
                         1); // Set to 1 when in1, in2 and out are ports from the same memory
    end
    
    8'd12: begin // [xz, xr] = sigm([recz, recr])
        instr_vecFunc(SIGM, // SIGM TANH ONEMA
                      MEMTEMPB, MEMTEMPA, // in, out
                      3*gru_out_size, 0, // address start in, out
                      2*gru_out_size, // number of positions
                      vecVecFunc_latency); // Set to 0 if data is ordered (which is most of the time)
    end
    
    8'd13: begin // [xr] = vecVecMult([xr], [htm1])
        instr_vecVecFunc(MULT, // MULT ADD
                         MEMTEMPB, MEMTEMPC, MEMTEMPA, // in1, in2, out
                         gru_out_size, 0, gru_out_size, // address start in1, in2, out
                         gru_out_size, // number of positions
                         0); // Set to 1 when in1, in2 and out are ports from the same memory
    end
    
    8'd14: begin // [rech] = vecMatMult(xr, gruRecKernel[. . 0])
        instr_vecMatMult(MEMPARAMS, MEMTEMPB, MEMTEMPA, // in1, in2, out
                         addr_st_grk + 2*memdepth_grk_individual, addr_st_grk + memdepth_grk, // address start/end in1 
                         gru_out_size, 2*gru_out_size, // address start/end in2 
                         3*gru_out_size, 4*gru_out_size); // address start/end out
    end
    
    8'd15: begin // [xh] = vecVecAdd([xh], [rech]) // All output data address is offset by a value which is equal to the latency of vecVecFunc (it wraps around)
        instr_vecVecFunc(ADD, // MULT MEMTEMPA
                         MEMTEMPA, MEMTEMPB, MEMTEMPA, // in1, in2, out
                         2*gru_out_size, 3*gru_out_size, 0, // address start in1, in2, out
                         gru_out_size, // number of positions
                         1); // Set to 1 when in1, in2 and out are ports from the same memory
    end
    
    8'd16: begin // [xh] = tanh([xh])
        instr_vecFunc(TANH, // SIGM TANH ONEMA
                      MEMTEMPB, MEMTEMPA, // in, out
                      2*gru_out_size, gru_out_size, // address start in, out
                      gru_out_size, // number of positions
                      vecVecFunc_latency); // Set to 0 if data is ordered (which is most of the time)
    end
    
    8'd17: begin // [1mz] = 1 - [xz]
        instr_vecFunc(ONEMA, // SIGM TANH ONEMA
                      MEMTEMPA, MEMTEMPC, // in, out
                      0, gru_out_size, // address start in, out
                      gru_out_size, // number of positions
                      0); // Set to 0 if data is ordered (which is most of the time)
    end
    
    8'd18: begin // [xz] = vecVecMult([xz], [htm1])
        instr_vecVecFunc(MULT, // MULT ADD
                         MEMTEMPB, MEMTEMPC, MEMTEMPA, // in1, in2, out
                         0, 0, 0, // address start in1, in2, out
                         gru_out_size, // number of positions
                         0); // Set to 1 when in1, in2 and out are ports from the same memory
    end
    
    8'd19: begin // [xh] = vecVecMult([1mz], [xh])
        instr_vecVecFunc(MULT, // MULT ADD
                         MEMTEMPB, MEMTEMPC, MEMTEMPA, // in1, in2, out
                         gru_out_size, gru_out_size, gru_out_size, // address start in1, in2, out
                         gru_out_size, // number of positions
                         0); // Set to 1 when in1, in2 and out are ports from the same memory
    end
    
    8'd20: begin // [htm1] = vecVecAdd([xz], [xh])
        instr_vecVecFunc(ADD, // MULT ADD
                         MEMTEMPA, MEMTEMPB, MEMTEMPC, // in1, in2, out
                         0, gru_out_size, 0, // address start in1, in2, out
                         gru_out_size, // number of positions
                         0); // Set to 1 when in1, in2 and out are ports from the same memory
    end
    
    8'd21: begin // jump to 8 if the last value for indata_offset hasn't been reached
        instr_type <= IT_SET_PC;

        if (addr_indata_offset > last_addr_indata_offset) begin
            next_prog_counter <= prog_counter + 1;
            addr_indata_offset <= feature_size; // reset to 'feature_size' (first value)
        end
        else
            next_prog_counter <= 8'd8;
    end
    
    // FC layer
    
    8'd22: begin // [out] = vecMatMult(htm1, fckernel)
        instr_vecMatMult(MEMPARAMS, MEMTEMPC, MEMTEMPA, // in1, in2, out
                         addr_st_fk, addr_st_fk + memdepth_fk, // address start/end in1 
                         0, gru_out_size, // address start/end in2 
                         0, memdepth_fb); // address start/end out
    end
    
    8'd23: begin // [out] = vecVecAdd([out], [fcbias])
        instr_vecVecFunc(ADD, // MULT ADD
                         MEMPARAMS, MEMTEMPB, MEMTEMPA, // in1, in2, out
                         addr_st_fb, 0, 0, // address start in1, in2, out
                         memdepth_fb, // number of positions
                         0); // Set to 1 when in1, in2 and out are ports from the same memory
    end
    
    8'd24: begin // outIndex = argmax(out)
        instr_type <= IT_MATH_OP;
        
        op_select <= ARGMAX;
    end
    
    default: begin
        //generally the default state is reached at the end of the program, we do nothing here
    end
endcase
