`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/08/2022 08:14:42 AM
// Design Name: 
// Module Name: simTest
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module simTest();

genvar i;

reg clk = 0;

/*
localparam memwidth = 64;
wire [memwidth-1:0]r_data;
wire [memwidth-1:0]r_data_in;

test dut(
    .clk(clk),
    .r_data(r_data),
    .r_data_in(r_data_in)
);
*/


/* // Counter test
wire [7:0]val0;
wire [7:0]val1;
reg rst = 1;
wire dut1en;

counter dut0(
    .clk(clk),
    .st_val(8'd0),
    .end_val(8'd10),
    .val(val0),
    .add_val(1),
    .en(1),
    .rst(rst),
    .valend(dut1en)
);

counter dut1(
    .clk(clk),
    .st_val(8'd10),
    .end_val(8'd50),
    .add_val(8'd10),
    .val(val1),
    .en(dut1en),
    .rst(rst)
);

wire [8:0] secondAddr = val0 + val1;

initial begin
    #10;
    rst = 0;
end
*/


reg rst = 1;

/*
reg st = 0;
wire rdy;

vecMatMult dut(
    .clk(clk),
    .rst(rst),
    .st(st),
    .rdy(rdy)
);
*/

/*
reg [7:0]tempRes[7:0];
reg [3:0]index = 0;

genvar i;
for (i = 0; i < 8; i = i+1) begin
    always @(posedge clk) begin
        if (rst) begin
            tempRes[i] <= 0;
        end
    end
end

wire tempResp1 = tempRes[index] + 1;
always @(posedge clk) begin
    tempRes[index] = tempResp1;
end
*/

localparam t = 5;

always begin
    clk = clk + 1; #(1*t);
end

reg st = 0;
reg rst = 1;
wire rdy;

reg ren_vecout = 0;
reg [16:0]r_addr_vecout = 0;
wire[64-1:0] r_data_vecout;

reg wen_vecin = 0;
reg[$clog2(250) - 1:0]w_addr_vecin = 0;
reg [64-1:0] w_data_vecin = 0;

/*
test dut(
    .clk(clk),
    .st(st),
    .rst(rst),
    
    .rdy(rdy),
    
    .ren_vecout(ren_vecout),
    .r_addr_vecout(r_addr_vecout),
    .r_data_vecout(r_data_vecout),
    
    .wen_indata(wen_vecin),
    .w_addr_indata(w_addr_vecin),
    .w_data_indata(w_data_vecin)
);
*/

localparam MEMPARAMS = 3'd0;
localparam MEMIN = 3'd1;
localparam MEMTEMPA = 3'd2;
localparam MEMTEMPB = 3'd3;
localparam MEMTEMPC = 3'd4;

localparam IN1 = 3'd0;
localparam IN2 = 3'd1;
localparam OUT = 3'd2;

reg [2:0]mem_select_in1 = MEMPARAMS; // MEMPARAMS
reg [2:0]mem_select_in2 = MEMIN; // MEMIN
reg [2:0]mem_select_out = MEMTEMPA; // MEMTEMPA

test2 dut(
    .clk(clk),
    .st(st),
    .rst(rst),
    
    .mem_select_in1(mem_select_in1),
    .mem_select_in2(mem_select_in2),
    .mem_select_out(mem_select_out),

    .mem_addr_in2(r_addr_vecout),
    .mem_r_data_in2(r_data_vecout),
    .mem_en_in2(ren_vecout),

    .rdy(rdy)
);


wire [7:0]vecOutArranged[7:0];
for (i = 0; i < 8; i=i+1) begin assign vecOutArranged[i] = r_data_vecout[i*(8)+:(8)]; end // flatten array

integer ii;

// ----- Verification
// Memory containing the result of vecMatMult
reg [63:0]pythonRes[149:0];
initial begin
    $readmemh("C:/Users/jdah1/Desktop/ASR/randomMultGruKernel.mem", pythonRes);
end

wire [16:0] #(t*2) r_addr_vecout_wd = r_addr_vecout;
wire [16:0] #(t*2) r_addr_vecout_wdd = r_addr_vecout_wd;
wire [16:0] #(t) r_addr_vecout_wdh = r_addr_vecout_wd;
wire [16:0] #(t*2) r_addr_vecout_wddd = r_addr_vecout_wdd;

wire [63:0]pythonRes_val = pythonRes[r_addr_vecout_wdh];

wire RES_COMPARISON = pythonRes_val == r_data_vecout;
// -----

initial begin
    #(t);
    rst = 1; #(t*10000);
    rst = 0; #(t*2);
    st = 1; #(t*2);
    st = 0;
    #(t*14000);
    mem_select_in2 = MEMTEMPA; // MEMTEMPA
    ren_vecout = 1; #(t*2);
    r_addr_vecout = 149; #(t*2);
    r_addr_vecout = 0; #(t*2);
    for (ii = 0; ii < 150; ii=ii+1) begin
        r_addr_vecout = r_addr_vecout + 1; #(t*2);
    end
    ren_vecout = 0; #(t*2);
    r_addr_vecout = 0; #(t*2);
    /*
    #(t*5);
    mem_select_in2 = MEMTEMPA;
    mem_select_out = MEMIN;
    st = 1; #(t*2);
    st = 0;
    */
end

endmodule
