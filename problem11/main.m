clear
close all

% training setting
stop_criteria = 1e-8;
max_iteration = 10000;
noise_strength = 1.0; % standard deviation of gaussian noise
M = 100;
N = 256;
F1 = 0.1;
F2 = 0.22;

% data
t = 0:M-1;
g = (2*cos(2*pi*F1*t) + sin(2*pi*F2*t))';
noise = noise_strength*randn(M, 1);
y = g + noise;

A = zeros(M, N);
for m = 1:M
    for n = 1:N
        A(m,n) = exp(1i*2*pi/N*(m-1)*(n-1)) / sqrt(N);
    end
end

% learning

% result

% view (original signal)
f1 = figure;
subplot(121), plot(1:M, g), title('True signal');
ylim([-5, 5]);
xlabel("m");
subplot(122), plot(1:M, y), title('Observed signal');
ylim([-5, 5]);
xlabel("m");
f1.Position(3:4) = [640 160];

% view (Fourier coefficients)
f = linspace(0, (N/2)/N, N/2);
fftg = abs(A'*g);
ffty = abs(A'*y);
f2 = figure;
subplot(121), stem(f, fftg(1:(N/2)), "MarkerSize", 3), title('FFT of True signal');
xlabel("Frequency");
subplot(122), stem(f, ffty(1:(N/2)), "MarkerSize", 3), title('FFT of Noisy signal');
xlabel("Frequency");
f2.Position(3:4) = [640 160];

% compare

% save
print('-f1', "true_signal_and_noisy_signal",'-dpng')
print('-f2', "true_signal_and_noisy_signal_FFT",'-dpng')
clear("f1")
clear("f2")
save("problem11_result")