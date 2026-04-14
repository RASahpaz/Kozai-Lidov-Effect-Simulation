function KL_Reverse_Migration_Sim()
    % --- Physical Constants (SI) ---
    G = 6.67430e-11; 
    MSun = 1.989e30; 
    MJup = 1.898e27; 
    RJup = 7.149e7;  
    AU = 1.496e11;   
    c = 2.9979e8;    
    
    % --- "Current" System State (Starting Point for Backwards) ---
    m1 = 0.6 * MSun;   
    m2 = 0.390 * MJup;   
    m3 = 0.31 * MSun;   
    
    % Assuming we found a planet with a 10-day period
    current_P_days = 10;
    target_P_sec = current_P_days * 24 * 3600;
    a_current = ((target_P_sec * sqrt(G*(m1+m2))) / (2*pi))^(2/3);
    
    e_current = 0.01;            % Circularized orbit
    inc_current = deg2rad(85);   % Current inclination (relative to binary)
    omega_current = deg2rad(90); % Argument of periastron
    
    a_out = 80 * AU;  
    e_out = 0.2; 
    tV2 = 0.1; 
    
    % --- Reverse Simulation Setup ---
    % t_span goes from 0 to -100 Myr
    t_end_backward = -100e8 * 31557600; 
    y_current = [a_current, e_current, inc_current, omega_current];
    
    % Stop if the planet reaches a likely birth semi-major axis (e.g., 5 AU)
    a_birth_limit = 5.0 * AU;
    options = odeset('RelTol', 1e-8, 'AbsTol', 1e-10, ...
                     'Events', @(t,y) stop_at_birth(t,y,a_birth_limit));
    
    fprintf('Running Time-Reversal Integration...\n');
    [t, y] = ode45(@(t, y) secular_dynamics_REVERSE(t, y, m1, m2, m3, a_out, e_out, tV2, RJup, G, c), ...
                   [0, t_end_backward], y_current, options);
               
    % --- Final Statistics (The "Past" State) ---
    t_myr_final = t(end)/(1e6 * 31557600);
    fprintf('Simulation went back to: %.2f Myr ago\n', abs(t_myr_final));
    fprintf('Likely original Semi-major Axis: %.4f AU\n', y(end,1)/AU);
    fprintf('Likely original Eccentricity: %.4f\n', y(end,2));
    fprintf('Likely original Inclination: %.2f degrees\n', rad2deg(y(end,3)));

    % --- Animated Visualization ---
    visualize_KL_animation(t, y, m1, m3, a_out, e_out, AU, G);
end

% --- Stop when the planet migrates "out" to its starting distance ---
function [value, isterminal, direction] = stop_at_birth(~, y, a_limit)
    value = y(1) - a_limit; 
    isterminal = 1;          
    direction = 1; % Trigger when 'a' is increasing (moving backwards)
end

function dydt = secular_dynamics_REVERSE(~, y, m1, m2, m3, a_out, e_out, tV2, RJup, G, c)
    a = y(1); e = y(2); inc = y(3); omega = y(4);
    
    % Keep eccentricity in physical bounds
    e = max(min(e, 0.9999), 1e-6); 
    
    Pin = 2*pi*sqrt(a^3 / (G*(m1+m2)));
    Pout = 2*pi*sqrt(a_out^3 / (G*(m1+m2+m3)));
    tau_KL = (2*Pout^2 / (3*pi*Pin)) * ((m1+m2+m3)/m3) * (1-e_out^2)^1.5;
    
    % Reversible Secular Dynamics (KL + GR)
    de_dt_KL = (15/8) * (e * sqrt(1-e^2) / tau_KL) * sin(inc)^2 * sin(2*omega);
    di_dt_KL = -(15/8) * (e^2 / (tau_KL * sqrt(1-e^2))) * sin(inc) * cos(inc) * sin(2*omega);
    domega_dt_KL = (3/(4*tau_KL*sqrt(1-e^2))) * (2*(1-e^2) + 5*sin(omega)^2*(e^2-sin(inc)^2));
    
    n = sqrt(G*(m1+m2)/a^3);
    domega_dt_GR = (3 * G * (m1 + m2) * n) / (a * c^2 * (1 - e^2));
    
    % Tidal Migration (Will be handled as "anti-dissipation" by the negative t_span)
    tV_sec = tV2 * 31557600; 
    tF2 = tV_sec * (9/8) * (a/RJup)^6 * (m1/m2); 
    
    V2 = (9/tF2) * (1 + (15/4)*e^2 + (15/8)*e^4 + (5/64)*e^6) / (1-e^2)^6.5;
    W2 = (1/tF2) * (1 + (15/2)*e^2 + (45/8)*e^4 + (5/16)*e^6) / (1-e^2)^6.5;
    
    da_dt_tide = -a * (2*V2 * (e^2/(1-e^2)) + 2*W2); 
    de_dt_tide = -e * (V2 + 9*W2); 
    
    dydt = [da_dt_tide; de_dt_KL + de_dt_tide; di_dt_KL; domega_dt_KL + domega_dt_GR];
end

function visualize_KL_animation(t, y, m1, m3, a_out, e_out, AU, G)
    % Use absolute value for time to plot 0 -> Past
    t_myr = abs(t / (1e6 * 31557600)); 
    fig = figure('Color', 'w', 'Position', [50, 50, 1300, 700]);
    
    subplot(3, 2, 1);
    plot(t_myr, y(:,1)/AU, 'b', 'LineWidth', 1.5);
    ylabel('a (AU)'); title('Semi-major Axis (Backward)'); grid on;
    h_a_mark = line(t_myr(1), y(1,1)/AU, 'Marker', 'o', 'Color', 'r', 'MarkerFaceColor', 'r');

    subplot(3, 2, 3);
    plot(t_myr, y(:,2), 'r', 'LineWidth', 1);
    ylabel('e'); title('Eccentricity (Backward)'); grid on;
    h_e_mark = line(t_myr(1), y(1,2), 'Marker', 'o', 'Color', 'b', 'MarkerFaceColor', 'b');
    
    subplot(3, 2, 5);
    P_out = 2*pi*sqrt(a_out^3 / (G*(m1+m3)));
    r_sep = a_out * (1 - e_out * cos(2 * pi * t / P_out));
    plot(t_myr, r_sep/AU, 'Color', [0.4 0.4 0.4], 'LineWidth', 1.5);
    ylabel('r_{sep} (AU)'); xlabel('Myr Ago'); title('Stellar Separation'); grid on;
    h_r_mark = line(t_myr(1), r_sep(1)/AU, 'Marker', 'o', 'Color', 'k', 'MarkerFaceColor', 'k');

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

    skip = max(1, round(length(t)/400)); 
    for i = 1:skip:length(t)
        if ~ishandle(fig), break; end
        phase = 2 * pi * (t(i) / P_out);
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
        set(h_planet, 'XData', (xp_rel+x1)/AU, ...
                      'YData', (yp_temp_rel*cos(y(i,3))+y1)/AU, ...
                      'ZData', (yp_temp_rel*sin(y(i,3)))/AU);
        
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