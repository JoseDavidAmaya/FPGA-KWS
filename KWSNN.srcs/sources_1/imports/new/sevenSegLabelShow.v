`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07/01/2022 08:40:07 PM
// Design Name: 
// Module Name: sevenSegLabelShow
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


module sevenSegLabelShow(
        input clk,
        input rst,
        input [3:0]idx,
        
        output [6:0]sevenSegment,
        output [7:0]segment
    );
    
    
    // clk_div to swap segment
    wire clk_div;
    counter #(19) clk_div_counter(
        // Inputs
        .clk(clk),
        .rst(rst),
        .en(1),
        
        .st_val(0),
        .end_val(19'd333_333), // 19'd333_333 // 19'd333
        .add_val(1),
        
        // Outputs
        .val(),
        .valend(clk_div)
    );
    
    // letter counter, swaps the segment based on clk_div
    wire [2:0]letterIdx;
    counter #(3) letter_counter(
        // Inputs
        .clk(clk), // TODO: Is it better to put it (the clk_div) here or on the enable? No, reset doesn't work
        .rst(rst),
        .en(clk_div),
        
        .st_val(0),
        .end_val(5),
        .add_val(1),
        
        // Outputs
        .val(letterIdx),
        .valend()
    );
    
    assign segment = ~(8'b1 << letterIdx);
    
    ////// ------ Testing
    /*
    wire clk_div_test;
    counter #(27) clk_div_test_counter(
        // Inputs
        .clk(clk),
        .rst(rst),
        .en(1),
        
        .st_val(0),
        .end_val(27'd100_000_000), //27'd100_000_000 // 27'd100_000
        .add_val(1),
        
        // Outputs
        .val(),
        .valend(clk_div_test)
    );
    
    wire [3:0]idx;
    counter #(4) word_sel_test_counter(
        // Inputs
        .clk(clk), // TODO: Is it better to put it (the clk_div) here or on the enable? No, reset doesn't work
        .rst(rst),
        .en(clk_div_test),
        
        .st_val(0),
        .end_val(12),
        .add_val(1),
        
        // Outputs
        .val(idx),
        .valend()
    );
    */
    ////// ------
    
    genvar i_idx, i_letterIdx;
    localparam numLabels = 12;
    localparam numLettersPerWord = 5;
    localparam numLetters = 17;
    reg [4:0]memlabels_reg[numLabels*numLettersPerWord-1:0];
    reg [7:0]memencoder[numLetters-1:0];
    
    wire [$clog2(numLabels*numLettersPerWord)-1:0]memlabels_reg_indices[numLabels-1:0][numLettersPerWord-1:0];
    for (i_idx = 0; i_idx < numLabels; i_idx = i_idx + 1)begin
        for (i_letterIdx = 0; i_letterIdx < numLettersPerWord; i_letterIdx = i_letterIdx + 1) begin
            assign memlabels_reg_indices[i_idx][i_letterIdx] = i_idx*numLettersPerWord + i_letterIdx;
        end
    end
    
    assign sevenSegment = memencoder[memlabels_reg[memlabels_reg_indices[idx][letterIdx]]];
    
    
    // Load encoder and labels memories
    initial begin
        $readmemh("C:/Users/jdah1/Desktop/ASR/mem/memlabels.mem", memlabels_reg);
        $readmemh("C:/Users/jdah1/Desktop/ASR/mem/memencoder.mem", memencoder);
    end
    
endmodule
