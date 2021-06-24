clear all;
close all;
clc;

Lx = 10;
Ly = 2;

Nelx=20;
Nely=4;
Nel = Nelx*Nely;

dx = Lx/Nelx;
dy = Ly/Nely;

Nnodes = (Nelx+1)*(Nely+1);
Nnodesx = Nelx+1;
Nnodesy = Nely+1;

x_vec = zeros(Nnodes,1);
y_vec = zeros(Nnodes,1);
z_vec = zeros(Nnodes,1);
xyz_mat = zeros(Nnodes,3);

vertex_ID = zeros(Nnodes,1);

M = zeros(Nel,4);

node_number = 1;
for i = 1:Nely+1
    for j = 1:Nelx+1
        
    x_vec(node_number,1) = (j-1)*dx;
    y_vec(node_number,1) = (i-1)*dy;
    z_vec(node_number,1) = 0;
    
    xyz_mat(node_number,1) = (j-1)*dx;
    xyz_mat(node_number,2) = (i-1)*dy;
    xyz_mat(node_number,3) = 0;
    
    vertex_ID(node_number,1) = node_number;
    
    node_number = node_number+1;
    
    end
end

el_number = 1;
for ii = 1:Nely
    for jj = 1:Nelx
        
        M(el_number,1)=(ii-1)*Nnodesx+jj;
        M(el_number,2)=(ii-1)*Nnodesx+jj+1;
        M(el_number,3)=(ii)*Nnodesx+jj+1;
        M(el_number,4)=(ii)*Nnodesx+jj;
        
        el_number=el_number+1;
        
    end
end

M_touse = M-1;