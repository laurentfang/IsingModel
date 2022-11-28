clear,clc

%% Initialization

J = 1;
h = 0; % External field
T = 10; % Temperature
N = 25; % Dimension of the system
iter = 2000000; % Number of Iteration
probSpin = 0.5; % Probability of up spin
system = sign(rand(N,N,N) - probSpin); % Create a spin matrix

%% Interpreting Monte Carlo Algorithm

monte_carlo(system,iter,N,h,J,T)

function monte_carlo(system,iter,N,h,J,T)

% Make visualization for 3D Monte Carlo Simulation
fig = figure;
slice(system,1,1,N)
colormap([1 0 0; 0 0 1]); 
xlim([1,N])
ylim([1,N])
zlim([1,N])
axis vis3d
camproj('perspective')
hold on

for n = 1:iter
    index = randi(N*N*N); % Randomly chosen the starting point
    [i,j,k] = ind2sub([N,N,N],index); % Calculate index's  3D coordinates
    rt = sub2ind([N,N,N],i,mod(j,N)+1,k); % Index for the right neighbor
    lt = sub2ind([N,N,N],i,mod(j-2,N)+1,k); 
    up = sub2ind([N,N,N],mod(i-2,N)+1,j,k); 
    dn = sub2ind([N,N,N],mod(i,N)+1,j,k); 
    ft = sub2ind([N,N,N],i,j,mod(k,N)+1); 
    bd = sub2ind([N,N,N],i,j,mod(k-2,N)+1); 

    neighbor = system(rt) + system(lt) + system(up) + system(dn) + system(ft) + system(bd);
    dE = 2*J*(system(index)*neighbor) + 2*h*system(index);

    prob = exp(-dE/T);

    if dE <= 0 || rand() <= prob
        % Metropolis Algorithm applied
        system(index) = -system(index);
    end

    freq = 2000; % Num of iteration for plot update
    
    if sum(ismember(1:freq:iter,n))
        % Plot the matrix at each f iteration
        slice(system,1,1,N) %plot the lattice in 3D matrix
        pause(0.00001)
    end
end

end
