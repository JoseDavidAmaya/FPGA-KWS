`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/07/2022 08:41:27 PM
// Design Name: 
// Module Name: sat
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

// Takes a number 'a' of n1 bits as input and outputs a number 'b' of n2 bits
// if a is greater than the maximum number b can represent (for n2 bits), then b takes the maximum value
// ... lesser ... minimum ...

`define signExtend(val, nIn, nOut) {{(nOut-nIn){val[nIn-1]}}, val}

module sat #(parameter n1=9, parameter n2=8)(
    input signed [n1-1:0]a,
    output signed [n2-1:0]b
    );
    
    /*
    wire signed [n1-1:0]maxVal = {{(n1-n2+1){1'b0}}, {(n2-1){1'b1}}};
    wire signed [n1-1:0]minVal = {{(n1-n2+1){1'b1}}, {(n2-1){1'b0}}};

    wire satMax = a > maxVal;
    wire satMin = a < minVal;
    
    assign b = satMax ? maxVal[n2-1:0] : (satMin ? minVal[n2-1:0] : a);
    */
    
    // Simpler version
    
    localparam inMSB = n1-1;
    localparam outMSB = n2-1;
    
    wire satMin = a[inMSB] & (~&a[inMSB-1:outMSB]);
    wire satMax = ~a[inMSB] & (|a[inMSB-1:outMSB]);
    
    wire signed [n2-1:0]maxVal = {{1'b0}, {(n2-1){1'b1}}};
    wire signed [n2-1:0]minVal = {{1'b1}, {(n2-1){1'b0}}};
    
    assign b = satMax ? maxVal[n2-1:0] : (satMin ? minVal[n2-1:0] : a);
    
endmodule
