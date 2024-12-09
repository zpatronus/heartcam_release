const video = document.getElementById('videoElement');
const ampVideo = document.getElementById('amplifiedVideoElement')
const canvas = document.getElementById('canvas');
const message = document.getElementById('message');
const info = document.getElementById('info');

const chartCanvas = document.getElementById('heartRateChart');
const pulseChartCanvas = document.getElementById('pulseChart');
let pulseHistory = [];
const pulseChartDuration = 10;

const context = canvas.getContext('2d');
const ampContext = ampVideo.getContext('2d');

function resizeCanvas () {
  ampVideo.width = video.videoWidth;
  ampVideo.height = video.videoHeight;
}

window.addEventListener('resize', resizeCanvas);

const bufferSize = 560;
// console.log('Buffer Size:', bufferSize);

const samplingTime = 10;
const abandonTime = 0;
const abandonPerFrame = 1
const abandonPerFrameAfterSamplingTime = 2
const freqMin = 0.66;
const freqMax = 3;
let signal = [];
let fingerDetected = false;
let countdownStartTime = null;
let smoothedHeartRate = null;
let popSignal = 0;
let smoothedBrightness = null;
let ampVideoBrightness = 0;
let lastPulseTime = Date.now();

let logs = []

function updateDataPointsInfo () {
  document.getElementById('dataPointsInfo').textContent = `Data points sampled: ${logs.length}`;
}

function addLogEntry (data) {
  logs.push({
    timestamp: new Date().toISOString(),
    ...data
  });
  updateDataPointsInfo();
}

function clearLog () {
  logs = [];
  updateDataPointsInfo();
}

function downloadLog () {
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(logs, null, 2));
  const downloadAnchorNode = document.createElement('a');
  downloadAnchorNode.setAttribute("href", dataStr);
  downloadAnchorNode.setAttribute("download", `heartcam_log_${new Date().toISOString()}.json`);
  document.body.appendChild(downloadAnchorNode);
  downloadAnchorNode.click();
  downloadAnchorNode.remove();
}

document.getElementById("clearLogButton").addEventListener("click", clearLog);
document.getElementById("downloadLogButton").addEventListener("click", downloadLog);


let fftChart = new Chart(chartCanvas, {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: 'Normalized Magnitude',
      data: [],
      borderColor: '#bb86fc',
      fill: false,
      pointStyle: 'circle',
      pointBackgroundColor: '#121212',
      pointRadius: 3,
      borderWidth: 2,
      tension: 0.3
    }]
  },
  options: {
    animation: {
      duration: 200
    },
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        type: 'linear',
        min: Math.round(freqMin * 60),
        max: Math.round(freqMax * 60),
        title: {
          display: true,
          text: 'Heart Rate (BPM)',
          color: 'rgba(187, 134, 252, 1)'
        },
        ticks: {
          stepSize: 10,
          color: 'rgba(187, 134, 252, 1)'
        },
        grid: {
          color: 'rgba(187, 134, 252, 0.2)'
        }
      },
      y: {
        min: 0,
        max: 1,
        title: {
          display: true,
          text: 'Normalized Magnitude',
          color: 'rgba(187, 134, 252, 1)'
        },
        grid: {
          color: 'rgba(187, 134, 252, 0.2)'
        },
        ticks: {
          color: 'rgba(187, 134, 252, 1)'
        }
      }
    },
    plugins: {
      legend: {
        display: false
      },
      annotation: {
        annotations: {}
      }
    }
  }
});

let pulseChart = new Chart(pulseChartCanvas, {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: 'Pulse History',
      data: [],
      borderColor: '#bb86fc',
      fill: false,
      pointRadius: 0,
      borderWidth: 2
    }]
  },
  options: {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        type: 'linear',
        min: 0,
        max: pulseChartDuration,
        title: {
          display: true,
          text: 'Time (s)',
          color: 'rgba(187, 134, 252, 1)'
        },
        ticks: {
          stepSize: 1,
          color: 'rgba(187, 134, 252, 1)',
          callback: function (value) {
            return (pulseChartDuration - value).toFixed(1);
          }
        },
        grid: {
          color: 'rgba(187, 134, 252, 0.2)',
          borderColor: 'rgba(187, 134, 252, 0.2)'
        }
      },
      y: {
        min: 0,
        max: 1,
        title: {
          display: true,
          text: 'Pulse Detected',
          color: 'rgba(187, 134, 252, 1)'
        },
        grid: {
          color: (context) => {
            return context.tick.value === 0 ? 'rgba(187, 134, 252, 0.2)' : 'transparent';
          },
        },
        ticks: {
          display: false
        }
      }
    },
    plugins: {
      legend: {
        display: false
      }
    }
  }
});



function updatePulseChart () {
  const currentTime = Date.now() / 1000;
  pulseHistory = pulseHistory.filter(p => p.time / 1000 > currentTime - pulseChartDuration);

  const labels = pulseHistory.map(p => (p.time / 1000 - currentTime + pulseChartDuration));
  const data = pulseHistory.map(p => p.value);

  pulseChart.data.labels = labels;
  pulseChart.data.datasets[0].data = data;
  pulseChart.update();
}

function startVideoStream () {
  navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } })
    .then(stream => {
      video.srcObject = stream;
      video.play();

      video.addEventListener('loadedmetadata', () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        requestAnimationFrame(processFrame);
      });
    })
    .catch(err => {
      console.warn('Front camera not available, trying rear camera.');
      navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
        .then(stream => {
          video.srcObject = stream;
          video.play();

          video.addEventListener('loadedmetadata', () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            requestAnimationFrame(processFrame);
          });
        })
        .catch(err => {
          message.textContent = 'Error accessing the camera: ' + err.name;
          message.style.color = 'red';
        });
    });
}

startVideoStream();

function findPeaks (minFreq, maxFreq, filteredFreqs, filteredMagnitudes) {
  let peakIndex = null;
  for (i = 0; i < filteredFreqs.length; ++i) {
    if (filteredFreqs[i] < minFreq) {
      continue;
    }
    if (filteredFreqs[i] > maxFreq) {
      break;
    }
    if (filteredMagnitudes[i] < 0.75) {
      continue;
    }
    if (peakIndex === null) {
      peakIndex = i;
    } else {
      if (filteredMagnitudes[i] > filteredMagnitudes[peakIndex]) {
        peakIndex = i;
      }
    }
  }
  return peakIndex
}

function processFrame () {
  if (video.videoWidth === 0 || video.videoHeight === 0) {
    requestAnimationFrame(processFrame);
    return;
  }

  resizeCanvas()

  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  const frame = context.getImageData(0, 0, canvas.width, canvas.height);

  const redDominancePercentage = extractRedDominance(frame);

  if (redDominancePercentage >= 99.95 && extractBrightnessDifference(frame) < 80) {
    // finger detected
    if (!fingerDetected) {
      fingerDetected = true;
      countdownStartTime = Date.now();
      signal = [];
    }

    const rgbAverage = extractRGBAverage(frame);
    const timestamp = Date.now();
    signal.push({ value: rgbAverage, time: timestamp / 1000 });

    if (signal.length > bufferSize) {
      signal.shift();
    }

    const timeDiff = signal[signal.length - 1].time - signal[0].time;
    const samplingRate = signal.length / (timeDiff || 1);
    // console.log('Sampling Rate:', samplingRate.toFixed(2), 'Hz, Buffer Size:', signal.length); 

    smoothedBrightness = smoothedBrightness === null ? rgbAverage : expSmooth(smoothedBrightness, rgbAverage, samplingRate, 20)
    if (rgbAverage > smoothedBrightness || lastPulseTime + 1 / (smoothedHeartRate / 60) * 2 / 3 * 1000 > Date.now()) {
      ampVideoBrightness = expSmooth(ampVideoBrightness, 0, samplingRate, 10)
    } else {
      ampVideoBrightness = 0.5
      lastPulseTime = timestamp
    }
    addLogEntry({ 'pulse': ampVideoBrightness / 0.5, 'bpm': smoothedHeartRate })
    pulseHistory.push({ time: timestamp, value: (ampVideoBrightness / 0.5) ** 3 });
    ampContext.fillStyle = `rgb(${187 * ampVideoBrightness}, ${134 * ampVideoBrightness}, ${252 * ampVideoBrightness})`;
    ampContext.fillRect(0, 0, ampVideo.width, ampVideo.height);
    updatePulseChart()


    info.innerHTML = `Sampling Rate: ${samplingRate.toFixed(2)} Hz, Buffer Size: ${signal.length}<br>GENTLY place your FINGERTIP on the camera, hold still, DO NOT move, and wait until the buffer size reaches ${bufferSize}.`;

    const elapsedTime = (Date.now() - countdownStartTime) / 1000;

    if (elapsedTime < samplingTime) {
      message.textContent = 'Hold still and cover the camera...';
      message.style.color = 'rgba(0, 255, 0, 1)';
    }
    if (elapsedTime > abandonTime) {
      if (popSignal >= (elapsedTime < samplingTime ? abandonPerFrame : abandonPerFrameAfterSamplingTime)) {
        signal.shift();
        popSignal = 0;
      } else {
        popSignal += 1;
      }
    }

    if (elapsedTime > Math.max(abandonTime, 1)) {
      // output heart rate
      const result = getFFT(signal, samplingRate);

      const filteredFreqs = result.freqs;
      const filteredMagnitudes = result.magnitudes;
      let tempPeakIndex = filteredMagnitudes.indexOf(Math.max(...filteredMagnitudes));
      let peakIndex = null;

      // test harmonics denoise
      // filteredMagnitudes[tempPeakIndex] = 0.8
      // for (let i = 0; i < filteredFreqs.length; ++i) {
      //   if (Math.abs(filteredFreqs[i] - 2 * filteredFreqs[tempPeakIndex]) < 5 / 60) {
      //     tempPeakIndex = i;
      //     break;
      //   }
      // }
      // filteredMagnitudes[tempPeakIndex] = 1

      if (filteredFreqs[tempPeakIndex] > 120.0 / 60) {
        peakIndex = findPeaks(filteredFreqs[tempPeakIndex] / 2.0 - 10.0 / 60, filteredFreqs[tempPeakIndex] / 2.0 + 10.0 / 60, filteredFreqs, filteredMagnitudes)
      }
      if (filteredFreqs[tempPeakIndex] < 50.0 / 60) {
        peakIndex = findPeaks(60.0 / 60, 100.0 / 60, filteredFreqs, filteredMagnitudes)
      }
      if (peakIndex === null) {
        peakIndex = tempPeakIndex;
      }

      let freqs = [filteredFreqs[peakIndex]]
      let mags = [filteredMagnitudes[peakIndex]]
      if (peakIndex > 0) {
        freqs.push(filteredFreqs[peakIndex - 1])
        mags.push(filteredMagnitudes[peakIndex - 1])
      }
      if (peakIndex < filteredFreqs.length - 1) {
        freqs.push(filteredFreqs[peakIndex + 1])
        mags.push(filteredMagnitudes[peakIndex + 1])
      }
      let magSum = mags.reduce((acc, mag) => acc + mag, 0);
      const heartRateBpm = 60 * freqs.reduce((acc, freq, index) => acc + freq * (mags[index] / magSum), 0);

      if (smoothedHeartRate === null || Number.isNaN(smoothedHeartRate)) {
        smoothedHeartRate = heartRateBpm;
      } else {
        smoothedHeartRate = expSmooth(smoothedHeartRate, heartRateBpm, samplingRate);
      }

      if (smoothedHeartRate === null || Number.isNaN(smoothedHeartRate)) {
        smoothedHeartRate = 70
      }

      updateChart(filteredFreqs.map(f => f * 60), filteredMagnitudes, smoothedHeartRate);

      message.textContent = `Heart Rate: ${smoothedHeartRate.toFixed(1)} bpm`;
      message.style.color = 'rgba(0, 255, 0, 1)';
    }
  } else {
    // finger not detected
    lastPulseTime = Date.now();
    fingerDetected = false;
    countdownStartTime = null;
    popSignal = 0;
    message.innerHTML = '1. Place your device and elbow on a stable surface.<br>2. GENTLY place your FINGERTIP on the camera. DO NOT PRESS HARD.<br>3. Hold still and COVER the camera with steady pressure...';
    message.style.color = 'rgba(187, 134, 252, 1)';
    info.textContent = '';
  }

  requestAnimationFrame(processFrame);
}

function extractRedDominance (frame) {
  const data = frame.data;
  let redDominantPixels = 0;

  for (let i = 0; i < data.length; i += 4) {
    const red = data[i];
    const green = data[i + 1];
    const blue = data[i + 2];

    if (red > green && red > blue) {
      redDominantPixels += 1;
    }
  }

  return (redDominantPixels / (data.length / 4)) * 100;
}

function extractRGBAverage (frame) {
  const data = frame.data;
  let total = 0;

  for (let i = 0; i < data.length; i += 4) {
    const red = data[i];
    const green = data[i + 1];
    const blue = data[i + 2];

    total += (red + green + blue) / 3;
  }

  return total / (data.length / 4);
}
function extractBrightnessDifference (frame) {
  const data = frame.data;
  let brightnessValues = [];
  const sampleRate = 0.01;

  for (let i = 0; i < data.length; i += (4 / sampleRate)) {
    const red = data[Math.floor(i)];
    const green = data[Math.floor(i) + 1];
    const blue = data[Math.floor(i) + 2];

    const brightness = (red + green + blue) / 3;
    brightnessValues.push(brightness);
  }

  function quickselect (arr, k) {
    if (arr.length === 1) return arr[0];

    const pivot = arr[Math.floor(arr.length / 2)];
    const lows = arr.filter(x => x < pivot);
    const highs = arr.filter(x => x > pivot);
    const pivots = arr.filter(x => x === pivot);

    if (k < lows.length) return quickselect(lows, k);
    else if (k < lows.length + pivots.length) return pivot;
    else return quickselect(highs, k - lows.length - pivots.length);
  }

  const pixelCount = brightnessValues.length;
  const percentile5 = quickselect(brightnessValues, Math.floor(0.05 * pixelCount));
  const percentile95 = quickselect(brightnessValues, Math.floor(0.95 * pixelCount));

  return (percentile95 - percentile5);
}

function expSmooth (currentRate, newRate, samplingRate, alpha = 1) {
  return 0.52 / samplingRate * alpha * newRate + (1 - 0.52 / samplingRate * alpha) * currentRate;
}

function getFFT (signal, samplingRate) {
  const values = signal.map(point => point.value);
  const n = values.length;

  // use the latest 2^n frames
  const size = Math.pow(2, Math.floor(Math.log2(n)));
  const startIdx = n - size;
  const latestValues = values.slice(startIdx, n);

  const mean = latestValues.reduce((a, b) => a + b, 0) / size;
  const zeroMeanValues = latestValues.map(v => v - mean);

  const fftResult = fft(zeroMeanValues);

  const frequencies = [];
  for (let i = 0; i < fftResult.length / 2; i++) {
    frequencies.push(i * samplingRate / fftResult.length);
  }

  const magnitudes = fftResult.slice(0, fftResult.length / 2).map(c => Math.sqrt(c.real * c.real + c.imag * c.imag));

  const filteredFreqs = [];
  const filteredMagnitudes = [];

  for (let i = 0; i < frequencies.length; i++) {
    if (frequencies[i] >= freqMin && frequencies[i] <= freqMax) {
      filteredFreqs.push(frequencies[i]);
      filteredMagnitudes.push(magnitudes[i]);
    }
  }

  // normalize magnitudes
  const maxMagnitude = Math.max(...filteredMagnitudes);
  const normalizedMagnitudes = filteredMagnitudes.map(m => m / maxMagnitude);

  return {
    freqs: filteredFreqs,
    magnitudes: normalizedMagnitudes
  };
}

// implement FFT function based on Cooley-Tukey algorithm
function fft (buffer) {
  const n = buffer.length;
  if (n <= 1) {
    return [{ real: buffer[0], imag: 0 }];
  }

  if ((n & (n - 1)) !== 0) {
    throw "FFT size must be a power of 2";
  }

  const half = n / 2;

  const even = new Array(half);
  const odd = new Array(half);
  for (let i = 0; i < half; i++) {
    even[i] = buffer[2 * i];
    odd[i] = buffer[2 * i + 1];
  }

  const evenFFT = fft(even);
  const oddFFT = fft(odd);

  const result = new Array(n);
  for (let k = 0; k < half; k++) {
    const angle = -2 * Math.PI * k / n;
    const t = { real: Math.cos(angle), imag: Math.sin(angle) };
    const oddComponent = multiplyComplex(t, oddFFT[k]);
    result[k] = addComplex(evenFFT[k], oddComponent);
    result[k + half] = subtractComplex(evenFFT[k], oddComponent);
  }

  return result;
}

function addComplex (a, b) {
  return { real: a.real + b.real, imag: a.imag + b.imag };
}

function subtractComplex (a, b) {
  return { real: a.real - b.real, imag: a.imag - b.imag };
}

function multiplyComplex (a, b) {
  return {
    real: a.real * b.real - a.imag * b.imag,
    imag: a.real * b.imag + a.imag * b.real
  };
}

function updateChart (freqs, magnitudes, smoothedHeartRate) {
  fftChart.data.labels = freqs;
  fftChart.data.datasets[0].data = magnitudes;

  fftChart.options.plugins.annotation = {
    annotations: {
      line1: {
        type: 'line',
        xMin: smoothedHeartRate,
        xMax: smoothedHeartRate,
        borderColor: 'rgba(0, 255, 0, 1)',
        borderWidth: 2,
        borderDash: [5, 5]
      },
      label1: {
        type: 'label',
        xValue: smoothedHeartRate + 14,
        yValue: 0.15,
        backgroundColor: 'rgba(18,18,18,0.8)',
        content: [`${smoothedHeartRate.toFixed(1)} BPM`],
        font: {
          size: 16
        },
        color: 'rgba(0, 255, 0, 1)'
      }
    }
  };

  fftChart.update();
}