`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07/02/2022 10:24:21 PM
// Design Name: 
// Module Name: spi_peripheral
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

// SPI Mode = 0
// Based on https://github.com/onchipuis/A-Connect/blob/main/demo_final/Verilog_files/spi_model.v and https://www.fpga4fun.com/SPI2.html

module spi_peripheral #(localparam n=8)(
    input clk,
    input rst,
    
    input ren, // enable reading, if 1, the next exchange, the data from data_miso will be send through MISO
    output reg data_mosi_ready, // means data_mosi data is valid and can be read
    
    input [n-1:0]data_miso,
    output [n-1:0]data_mosi,
    
    input SCK,
    input MOSI,
    output MISO,
    input CS
    );
    
    // Sync inputs using registers
    
    localparam REGS = 3; // 2 or 3
    
    (* ASYNC_REG="TRUE" *) reg [REGS-1:0]SCK_sync;
    (* ASYNC_REG="TRUE" *) reg [REGS-1:0]MOSI_sync;
    (* ASYNC_REG="TRUE" *) reg [REGS-1:0]CS_sync;
    always @(posedge clk) begin
        if (rst)
            SCK_sync <= 0;
        else 
            SCK_sync <= {SCK_sync[REGS-2:0], SCK};
            
        MOSI_sync <= {MOSI_sync[REGS-2:0], MOSI};
        CS_sync <= {CS_sync[REGS-2:0], CS};
    end
    
    wire SCK_RE = SCK_sync[REGS-1:REGS-2] == 2'b01; // Rising edge
    wire SCK_FE = SCK_sync[REGS-1:REGS-2] == 2'b10; // Falling edge
    
    wire CS_active = ~rst & ~CS_sync[REGS-2];

    //// Reading and writing
    reg [2:0]r_bit_count; // number of bits that have been read
    reg [n-1:0]MOSI_SR;
    reg [n-1:0]MISO_SR;
    always @(posedge clk) begin
        if(~CS_active) begin
            r_bit_count <= 0;
            if (ren)
                MISO_SR <= data_miso;
        end
        else begin
            if (SCK_RE) begin // Read on rising edge
                r_bit_count <= r_bit_count + 1;
                MOSI_SR <= {MOSI_SR[n-2:0], MOSI_sync[REGS-2]};
            end
        
            if (SCK_FE & ren) begin // Write on falling edge
                MISO_SR <= {MISO_SR[n-2:0], 1'b1};
            end
        end
    end
    
    assign data_mosi = MOSI_SR;
    assign MISO = ren ? MISO_SR[n-1] : 1'b1;
    
    always @(posedge clk) data_mosi_ready <= CS_active & SCK_RE & (&r_bit_count);
    
endmodule
