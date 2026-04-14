function KL_Migration_Sim()
    % --- Physical Constants (SI) ---
    G = 6.67430e-11; 
    MSun = 1.989e30; 
    MJup = 1.898e27; 
    RJup = 7.149e7;  
    AU = 1.496e11;   
    c = 2.9979e8;    
    
    % --- System Configuration ---
    m1 = 0.6 * MSun;   
    m2 = 0.390 * MJup;   
    m3 = 0.31 * MSun;   
    
    a_in = 5.0 * AU;         
    e_in = 0.128;             
    inc_tot = deg2rad(87.05); 
    omega_in = 0;            
    
    a_out = 80 * AU;  
    e_out = 0.2; % Added eccentricity to make the distance graph interesting
    
    tV2 = 0.1; 
    
    % --- Simulation Setup ---
    t_end = 2e6 * 31557600; 
    y0 = [a_in, e_in, inc_tot, omega_in];
    options = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
    
    fprintf('Simulating evolution...\n');
    [t, y] = ode45(@(t, y) secular_dynamics(t, y, m1, m2, m3, a_out, e_out, tV2, RJup, G, c), ...
                   [0, t_end], y0, options);
               
    % --- Animated Visualization with Companion Distance Graph ---
    visualize_KL_animation(t, y, m1, m3, a_out, e_out, AU, G);
end

function dydt = secular_dynamics(~, y, m1, m2, m3, a_out, e_out, tV2, RJup, G, c)
    a = y(1); e = y(2); inc = y(3); omega = y(4);
    
    Pin = 2*pi*sqrt(a^3 / (G*(m1+m2)));
    Pout = 2*pi*sqrt(a_out^3 / (G*(m1+m2+m3)));
    tau_KL = (2*Pout^2 / (3*pi*Pin)) * ((m1+m2+m3)/m3) * (1-e_out^2)^1.5;
    
    de_dt_KL = (15/8) * (e * sqrt(1-e^2) / tau_KL) * sin(inc)^2 * sin(2*omega);
    di_dt_KL = -(15/8) * (e^2 / (tau_KL * sqrt(1-e^2))) * sin(inc) * cos(inc) * sin(2*omega);
    domega_dt_KL = (3/(4*tau_KL*sqrt(1-e^2))) * (2*(1-e^2) + 5*sin(omega)^2*(e^2-sin(inc)^2));
    
    n = sqrt(G*(m1+m2)/a^3);
    domega_dt_GR = (3 * G * (m1 + m2) * n) / (a * c^2 * (1 - e^2));
    
    tV_sec = tV2 * 31557600; 
    tF2 = tV_sec * (9/8) * (a/RJup)^6 * (m1/m2); 
    
    V2 = (9/tF2) * (1 + (15/4)*e^2 + (15/8)*e^4 + (5/64)*e^6) / (1-e^2)^6.5;
    W2 = (1/tF2) * (1 + (15/2)*e^2 + (45/8)*e^4 + (5/16)*e^6) / (1-e^2)^6.5;
    
    da_dt_tide = -a * (2*V2 * (e^2/(1-e^2)) + 2*W2); 
    de_dt_tide = -e * (V2 + 9*W2); 
    
    dydt = [da_dt_tide; de_dt_KL + de_dt_tide; di_dt_KL; domega_dt_KL + domega_dt_GR];
end

function visualize_KL_animation(t, y, m1, m3, a_out, e_out, AU, G)
    t_myr = t / (1e6 * 31557600);
    fig = figure('Color', 'w', 'Position', [50, 50, 1300, 700]);
    
    % 1. Semi-major Axis (Top Left)
    subplot(3, 2, 1);
    plot(t_myr, y(:,1)/AU, 'b', 'LineWidth', 1.5);
    ylabel('a (AU)'); title('Planet Semi-major Axis'); grid on;
    h_a_mark = line(t_myr(1), y(1,1)/AU, 'Marker', 'o', 'Color', 'r', 'MarkerFaceColor', 'r');

    % 2. Planet Eccentricity (Middle Left)
    subplot(3, 2, 3);
    plot(t_myr, y(:,2), 'r', 'LineWidth', 1);
    ylabel('e'); title('Planet Eccentricity'); grid on;
    h_e_mark = line(t_myr(1), y(1,2), 'Marker', 'o', 'Color', 'b', 'MarkerFaceColor', 'b');
    
    % 3. Companion Distance Graph (Bottom Left)
    subplot(3, 2, 5);
    P_out = 2*pi*sqrt(a_out^3 / (G*(m1+m3)));
    % Instantaneous distance for eccentric orbit: r = a(1-e*cos(E))
    % Using mean anomaly (M) as an approximation for E for visualization
    r_sep = a_out * (1 - e_out * cos(2 * pi * t / P_out));
    plot(t_myr, r_sep/AU, 'Color', [0.4 0.4 0.4], 'LineWidth', 1.5);
    ylabel('r_{sep} (AU)'); xlabel('Time (Myr)'); title('Stellar Separation'); grid on;
    h_r_mark = line(t_myr(1), r_sep(1)/AU, 'Marker', 'o', 'Color', 'k', 'MarkerFaceColor', 'k');

    % 4. 3D Animation (Right)
    ax3 = subplot(3, 2, [2, 4, 6]);
    hold on; grid on; axis equal;
    
    a1 = a_out * (m3 / (m1 + m3)); 
    a3 = a_out * (m1 / (m1 + m3)); 

    h_star1  = plot3(0, 0, 0, 'yo', 'MarkerSize', 10, 'MarkerFaceColor', 'y', 'DisplayName', 'Host Star');
    h_star3  = plot3(0, 0, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', [0.5 0.5 0.5], 'DisplayName', 'Companion');
    h_orbit  = plot3(0, 0, 0, 'k-', 'LineWidth', 1.2); 
    h_planet = plot3(0, 0, 0, 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'r'); 
    
    xlabel('X (AU)'); ylabel('Y (AU)'); zlabel('Z (AU)');
    view(45, 30);
    limit = (a3/AU) * 1.5;
    axis(ax3, [-limit limit -limit limit -limit/2 limit/2]);

    % Animation loop
    skip = max(1, round(length(t)/350)); 
    for i = 1:skip:length(t)
        if ~ishandle(fig), break; end
        
        phase = 2 * pi * (t(i) / P_out);
        % Current separation factor for eccentric orbit
        r_fact = (1 - e_out * cos(phase)); 
        
        x1 = -a1 * r_fact * cos(phase); y1 = -a1 * r_fact * sin(phase);
        x3 =  a3 * r_fact * cos(phase); y3 =  a3 * r_fact * sin(phase);
        
        set(h_star1, 'XData', x1/AU, 'YData', y1/AU);
        set(h_star3, 'XData', x3/AU, 'YData', y3/AU);
        
        [X, Y, Z] = get_orbit_path(y(i,1), y(i,2), y(i,3), y(i,4));
        set(h_orbit, 'XData', (X+x1)/AU, 'YData', (Y+y1)/AU, 'ZData', Z/AU);
        
        r_p = (y(i,1)*(1-y(i,2)));
        xp_rel = r_p * cos(y(i,4));
        yp_temp_rel = r_p * sin(y(i,4));
        set(h_planet, 'XData', (xp_rel+x1)/AU, 'YData', (yp_temp_rel*cos(y(i,3))+y1)/AU, 'ZData', (yp_temp_rel*sin(y(i,3)))/AU);
        
        set(h_a_mark, 'XData', t_myr(i), 'YData', y(i,1)/AU);
        set(h_e_mark, 'XData', t_myr(i), 'YData', y(i,2));
        set(h_r_mark, 'XData', t_myr(i), 'YData', r_sep(i)/AU);
        
        drawnow; 
    end
end

function [x, y, z] = get_orbit_path(a, e, inc, omega)
    theta = linspace(0, 2*pi, 100);
    r = a * (1 - e^2) ./ (1 + e * cos(theta));
    x_orb = r .* cos(theta);
    y_orb = r .* sin(theta);
    x = x_orb * cos(omega) - y_orb * sin(omega);
    y_temp = x_orb * sin(omega) + y_orb * cos(omega);
    y = y_temp * cos(inc);
    z = y_temp * sin(inc);
end