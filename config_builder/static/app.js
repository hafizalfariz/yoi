/* ============================================
   YOI Config Builder - Frontend Application
   ============================================ */

// ===== STATE =====
var state = {
  feature: null,
  featureParams: {},
  sourceMode: 'inference',
  drawMode: null, // 'region' or 'line'
  currentPoints: [],
  selectedColor: '#0080ff',
  regions: [],
  lines: [],
  regionIdCounter: 1,
  lineIdCounter: 1,
  backgroundImage: null,  // Store loaded image
  
  // Line direction picker state
  lineDirectionPickerActive: false,
  lineDirectionPickerLine: null,
  lineDirectionHoveredArea: null  // 'above' or 'below'
};

// ===== COLOR PALETTE =====
var colors = [
  '#0080ff', '#ff8800', '#ff0066', '#00ff88',
  '#ffff00', '#8800ff', '#00ffff', '#ff0000'
];

// ===== CANVAS =====
var canvas = document.getElementById('canvas');
var ctx = canvas.getContext('2d');

// ===== FEATURE DEFINITIONS =====
var featureDefinitions = {
  region_crowd: {
    alert_threshold: { 
      type: 'number', 
      label: 'Alert Threshold', 
      default: 5,
      description: 'Jumlah orang dalam region untuk trigger alert (misal: alert jika ada 5+ orang)'
    },
    cooldown: { 
      type: 'number', 
      label: 'Cooldown (seconds)', 
      default: 300,
      description: 'Jeda waktu sebelum alert berikutnya (misal: 300 detik = 5 menit)'
    },
    warning_threshold: { 
      type: 'number', 
      label: 'Warning Threshold', 
      default: 5,
      description: 'Jumlah orang untuk warning level (peringatan ringan)'
    },
    critical_threshold: { 
      type: 'number', 
      label: 'Critical Threshold', 
      default: 10,
      description: 'Jumlah orang untuk critical level (peringatan serius)'
    },
    time_allowed: {
      type: 'time_range',
      label: 'Time Allowed (Schedule)',
      default: '',
      description: 'Jam operasional AI (misal: 07:00:00-17:00:00 untuk jam 7 pagi - 5 sore). Aktifkan checkbox untuk set jadwal'
    },
    tracking_tracker_impl: {
      type: 'select',
      label: 'Tracker Implementation',
      default: 'bytetrack',
      options: ['bytetrack', 'centroid'],
      statePath: 'tracking.tracker_impl',
      description: 'Pilih tracker utama. Rekomendasi: bytetrack'
    },
    tracking_bt_track_high_thresh: {
      type: 'number',
      label: 'ByteTrack High Threshold',
      default: 0.5,
      statePath: 'tracking.bt_track_high_thresh',
      description: 'Confidence tinggi untuk asosiasi track utama'
    },
    tracking_bt_track_low_thresh: {
      type: 'number',
      label: 'ByteTrack Low Threshold',
      default: 0.1,
      statePath: 'tracking.bt_track_low_thresh',
      description: 'Confidence rendah untuk asosiasi tahap kedua'
    },
    tracking_bt_new_track_thresh: {
      type: 'number',
      label: 'ByteTrack New Track Threshold',
      default: 0.6,
      statePath: 'tracking.bt_new_track_thresh',
      description: 'Minimum confidence untuk membuat track baru'
    },
    tracking_bt_match_thresh: {
      type: 'number',
      label: 'ByteTrack Match Threshold',
      default: 0.8,
      statePath: 'tracking.bt_match_thresh',
      description: 'Threshold matching antar frame (lebih besar = lebih ketat)'
    },
    tracking_bt_track_buffer: {
      type: 'number',
      label: 'ByteTrack Track Buffer',
      default: 45,
      statePath: 'tracking.bt_track_buffer',
      description: 'Jumlah frame buffer untuk mempertahankan track'
    },
    tracking_bt_fuse_score: {
      type: 'checkbox',
      label: 'ByteTrack Fuse Score',
      default: true,
      statePath: 'tracking.bt_fuse_score',
      description: 'Gabungkan skor deteksi pada proses matching'
    },
    tracking_reid_enabled: {
      type: 'checkbox',
      label: 'Enable ReID',
      default: true,
      statePath: 'tracking.reid_enabled',
      description: 'Aktifkan appearance matching untuk stabilisasi ID saat occlusion/silang'
    },
    tracking_reid_similarity_thresh: {
      type: 'number',
      label: 'ReID Similarity Threshold',
      default: 0.86,
      statePath: 'tracking.reid_similarity_thresh',
      description: 'Batas minimal similarity untuk recovery ID (0-1)'
    },
    tracking_reid_momentum: {
      type: 'number',
      label: 'ReID Embedding Momentum',
      default: 0.25,
      statePath: 'tracking.reid_momentum',
      description: 'EMA update embedding track (0-1); lebih kecil = lebih stabil'
    }
  },
  line_cross: {
    centroid: { 
      type: 'select', 
      label: 'Centroid (Detection Point)', 
      default: 'mid_centre',
      options: ['mid_centre', 'head', 'bottom'],
      description: 'Body part for detection: mid_centre (center), head (kepala), bottom (kaki)'
    },
    lost_threshold: { 
      type: 'number', 
      label: 'Lost Threshold (frames)', 
      default: 10,
      description: 'Frames tolerance sebelum tracking reset'
    },
    allow_recounting: { 
      type: 'checkbox', 
      label: 'Allow Recounting', 
      default: true,
      description: 'Prevent double counting same person/vehicle'
    },
    time_allowed: {
      type: 'time_range',
      label: 'Time Allowed (Schedule)',
      default: '',
      description: 'Jam operasional AI (misal: 07:00:00-17:00:00 untuk jam 7 pagi - 5 sore). Aktifkan checkbox untuk set jadwal'
    },
    tracking_tracker_impl: {
      type: 'select',
      label: 'Tracker Implementation',
      default: 'bytetrack',
      options: ['bytetrack', 'centroid'],
      statePath: 'tracking.tracker_impl',
      description: 'Pilih tracker utama. Rekomendasi: bytetrack'
    },
    tracking_bt_track_high_thresh: {
      type: 'number',
      label: 'ByteTrack High Threshold',
      default: 0.5,
      statePath: 'tracking.bt_track_high_thresh',
      description: 'Confidence tinggi untuk asosiasi track utama'
    },
    tracking_bt_track_low_thresh: {
      type: 'number',
      label: 'ByteTrack Low Threshold',
      default: 0.1,
      statePath: 'tracking.bt_track_low_thresh',
      description: 'Confidence rendah untuk asosiasi tahap kedua'
    },
    tracking_bt_new_track_thresh: {
      type: 'number',
      label: 'ByteTrack New Track Threshold',
      default: 0.6,
      statePath: 'tracking.bt_new_track_thresh',
      description: 'Minimum confidence untuk membuat track baru'
    },
    tracking_bt_match_thresh: {
      type: 'number',
      label: 'ByteTrack Match Threshold',
      default: 0.8,
      statePath: 'tracking.bt_match_thresh',
      description: 'Threshold matching antar frame (lebih besar = lebih ketat)'
    },
    tracking_bt_track_buffer: {
      type: 'number',
      label: 'ByteTrack Track Buffer',
      default: 45,
      statePath: 'tracking.bt_track_buffer',
      description: 'Jumlah frame buffer untuk mempertahankan track'
    },
    tracking_bt_fuse_score: {
      type: 'checkbox',
      label: 'ByteTrack Fuse Score',
      default: true,
      statePath: 'tracking.bt_fuse_score',
      description: 'Gabungkan skor deteksi pada proses matching'
    },
    tracking_reid_enabled: {
      type: 'checkbox',
      label: 'Enable ReID',
      default: true,
      statePath: 'tracking.reid_enabled',
      description: 'Aktifkan appearance matching untuk stabilisasi ID saat occlusion/silang'
    },
    tracking_reid_similarity_thresh: {
      type: 'number',
      label: 'ReID Similarity Threshold',
      default: 0.86,
      statePath: 'tracking.reid_similarity_thresh',
      description: 'Batas minimal similarity untuk recovery ID (0-1)'
    },
    tracking_reid_momentum: {
      type: 'number',
      label: 'ReID Embedding Momentum',
      default: 0.25,
      statePath: 'tracking.reid_momentum',
      description: 'EMA update embedding track (0-1); lebih kecil = lebih stabil'
    }
  },
  dwell_time: {
    alert_threshold: { 
      type: 'number', 
      label: 'Alert Threshold (seconds)', 
      default: 10,
      description: 'Waktu diam (detik) untuk trigger alert (misal: alert jika diam 10+ detik)'
    },
    cooldown: { 
      type: 'number', 
      label: 'Cooldown (seconds)', 
      default: 120,
      description: 'Jeda waktu sebelum alert berikutnya (misal: 120 detik = 2 menit)'
    },
    warning_seconds: { 
      type: 'number', 
      label: 'Warning Seconds', 
      default: 10,
      description: 'Durasi diam untuk warning level (peringatan ringan)'
    },
    critical_seconds: { 
      type: 'number', 
      label: 'Critical Seconds', 
      default: 20,
      description: 'Durasi diam untuk critical level (peringatan serius)'
    },
    min_dwelltime: { 
      type: 'number', 
      label: 'Min Dwell Time', 
      default: 15,
      description: 'Minimum waktu diam yang dihitung (detik) - mengabaikan yang lebih singkat'
    },
    lost_threshold: { 
      type: 'number', 
      label: 'Lost Threshold', 
      default: 10,
      description: 'Maksimal frame hilang sebelum tracking dianggap lost (reset counter)'
    },
    time_allowed: {
      type: 'time_range',
      label: 'Time Allowed (Schedule)',
      default: '',
      description: 'Jam operasional AI (misal: 07:00:00-17:00:00 untuk jam 7 pagi - 5 sore). Aktifkan checkbox untuk set jadwal'
    },
    tracking_tracker_impl: {
      type: 'select',
      label: 'Tracker Implementation',
      default: 'bytetrack',
      options: ['bytetrack', 'centroid'],
      statePath: 'tracking.tracker_impl',
      description: 'Pilih tracker utama. Rekomendasi: bytetrack'
    },
    tracking_bt_track_high_thresh: {
      type: 'number',
      label: 'ByteTrack High Threshold',
      default: 0.5,
      statePath: 'tracking.bt_track_high_thresh',
      description: 'Confidence tinggi untuk asosiasi track utama'
    },
    tracking_bt_track_low_thresh: {
      type: 'number',
      label: 'ByteTrack Low Threshold',
      default: 0.1,
      statePath: 'tracking.bt_track_low_thresh',
      description: 'Confidence rendah untuk asosiasi tahap kedua'
    },
    tracking_bt_new_track_thresh: {
      type: 'number',
      label: 'ByteTrack New Track Threshold',
      default: 0.6,
      statePath: 'tracking.bt_new_track_thresh',
      description: 'Minimum confidence untuk membuat track baru'
    },
    tracking_bt_match_thresh: {
      type: 'number',
      label: 'ByteTrack Match Threshold',
      default: 0.8,
      statePath: 'tracking.bt_match_thresh',
      description: 'Threshold matching antar frame (lebih besar = lebih ketat)'
    },
    tracking_bt_track_buffer: {
      type: 'number',
      label: 'ByteTrack Track Buffer',
      default: 60,
      statePath: 'tracking.bt_track_buffer',
      description: 'Jumlah frame buffer untuk mempertahankan track'
    },
    tracking_bt_fuse_score: {
      type: 'checkbox',
      label: 'ByteTrack Fuse Score',
      default: true,
      statePath: 'tracking.bt_fuse_score',
      description: 'Gabungkan skor deteksi pada proses matching'
    },
    tracking_reid_enabled: {
      type: 'checkbox',
      label: 'Enable ReID',
      default: true,
      statePath: 'tracking.reid_enabled',
      description: 'Aktifkan appearance matching untuk stabilisasi ID saat occlusion/silang'
    },
    tracking_reid_similarity_thresh: {
      type: 'number',
      label: 'ReID Similarity Threshold',
      default: 0.86,
      statePath: 'tracking.reid_similarity_thresh',
      description: 'Batas minimal similarity untuk recovery ID (0-1)'
    },
    tracking_reid_momentum: {
      type: 'number',
      label: 'ReID Embedding Momentum',
      default: 0.25,
      statePath: 'tracking.reid_momentum',
      description: 'EMA update embedding track (0-1); lebih kecil = lebih stabil'
    }
  }
};

// ===== INITIALIZATION =====
function init() {
  log('INIT', 'Application starting...');
  
  setupColorPalette();
  setupFeatureSelection();
  setupSourceMode();
  setupLogsConfig();
  setupImageUpload();
  setupCanvas();
  setupDrawingControls();
  setupLinePropertiesEditor();
  setupBuildControls();
  setupLoadConfig();
  loadLogs();
  
  log('INIT', 'Application ready!');
}

function setupLinePropertiesEditor() {
  var editor = document.getElementById('linePropertiesEditor');
  if (!editor) return;

  var btnSave = document.getElementById('btnSaveLineProps');
  var btnCancel = document.getElementById('btnCancelLineProps');

  if (btnSave) btnSave.onclick = function() { saveLineProperties(); };
  if (btnCancel) btnCancel.onclick = function() { closeLinePropertiesEditor(); };
}

// ===== COLOR PALETTE =====
function setupColorPalette() {
  var palette = document.getElementById('colorPalette');
  
  colors.forEach(function(color) {
    var chip = document.createElement('div');
    chip.className = 'color-chip';
    chip.style.backgroundColor = color;
    chip.onclick = function() {
      selectColor(color);
    };
    palette.appendChild(chip);
  });
  
  selectColor(colors[0]);
}

function selectColor(color) {
  state.selectedColor = color;
  
  var chips = document.querySelectorAll('.color-chip');
  chips.forEach(function(chip) {
    chip.classList.remove('active');
    if (chip.style.backgroundColor === hexToRgb(color)) {
      chip.classList.add('active');
    }
  });
  
  log('COLOR', 'Selected color: ' + color);
}

function hexToRgb(hex) {
  var r = parseInt(hex.slice(1, 3), 16);
  var g = parseInt(hex.slice(3, 5), 16);
  var b = parseInt(hex.slice(5, 7), 16);
  return 'rgb(' + r + ', ' + g + ', ' + b + ')';
}

// ===== FEATURE SELECTION =====
function setupFeatureSelection() {
  var cards = document.querySelectorAll('.feature-card');
  
  cards.forEach(function(card) {
    card.onclick = function() {
      var feature = card.getAttribute('data-feature');
      selectFeature(feature);
    };
  });
}

function selectFeature(feature) {
  state.feature = feature;
  
  // Visual feedback
  var cards = document.querySelectorAll('.feature-card');
  cards.forEach(function(card) {
    card.classList.remove('active');
    if (card.getAttribute('data-feature') === feature) {
      card.classList.add('active');
    }
  });
  
  // Show parameters
  renderFeatureParams(feature);
  
  log('FEATURE', 'Selected: ' + feature);
}

function renderFeatureParams(feature) {
  var container = document.getElementById('paramsContainer');
  var paramsDiv = document.getElementById('featureParams');
  
  if (!feature || !featureDefinitions[feature]) {
    paramsDiv.classList.add('hidden');
    return;
  }
  
  container.innerHTML = '';
  var params = featureDefinitions[feature];
  var trackingHeaderInserted = false;
  var advancedTrackingHeaderInserted = false;
  var keys = Object.keys(params)
    .map(function(key, index) {
      return { key: key, index: index };
    })
    .sort(function(a, b) {
      var pa = getParamPriority(a.key);
      var pb = getParamPriority(b.key);
      if (pa !== pb) return pa - pb;
      return a.index - b.index;
    })
    .map(function(item) {
      return item.key;
    });

  var overview = buildFeatureOverview(params, keys);
  if (overview) {
    container.appendChild(overview);
  }
  
  keys.forEach(function(key) {
    var param = params[key];
    var statePath = param.statePath || key;
    var inputIdSuffix = statePath.replace(/\./g, '__');

    if (key.indexOf('tracking_') === 0 && !trackingHeaderInserted) {
      var trackingHeader = document.createElement('div');
      trackingHeader.textContent = 'Tracking';
      trackingHeader.style.margin = '20px 0 10px 0';
      trackingHeader.style.paddingTop = '10px';
      trackingHeader.style.borderTop = '1px solid var(--border)';
      trackingHeader.style.fontSize = '13px';
      trackingHeader.style.fontWeight = '600';
      trackingHeader.style.color = 'var(--muted)';
      trackingHeader.style.letterSpacing = '0.02em';
      container.appendChild(trackingHeader);
      trackingHeaderInserted = true;
    }

    if (
      (key === 'tracking_bt_track_high_thresh' ||
        key === 'tracking_bt_track_low_thresh' ||
        key === 'tracking_bt_new_track_thresh' ||
        key === 'tracking_bt_match_thresh') &&
      !advancedTrackingHeaderInserted
    ) {
      var advancedHeader = document.createElement('div');
      advancedHeader.textContent = 'Advanced';
      advancedHeader.style.margin = '12px 0 8px 0';
      advancedHeader.style.fontSize = '12px';
      advancedHeader.style.fontWeight = '600';
      advancedHeader.style.color = 'var(--muted)';
      advancedHeader.style.letterSpacing = '0.02em';
      container.appendChild(advancedHeader);
      advancedTrackingHeaderInserted = true;
    }

    var group = document.createElement('div');
    group.className = 'form-group';
    group.style.marginBottom = '16px';
    
    var label = document.createElement('label');
    label.textContent = param.label;
    
    var input;
    
    // Create different input types based on param.type
    if (param.type === 'select') {
      input = document.createElement('select');
      input.id = 'param_' + inputIdSuffix;
      input.style.padding = '8px';
      input.style.borderRadius = '4px';
      input.style.border = '1px solid var(--border)';
      input.style.background = 'var(--panel-2)';
      input.style.color = 'var(--text)';
      
      param.options.forEach(function(option) {
        var opt = document.createElement('option');
        opt.value = option;
        opt.textContent = option;
        opt.style.background = 'var(--panel-2)';
        opt.style.color = 'var(--text)';
        if (option === param.default) {
          opt.selected = true;
        }
        input.appendChild(opt);
      });
      
      input.onchange = function() {
        setStateParam(statePath, input.value);
      };
      setStateParam(statePath, param.default);
      
    } else if (param.type === 'checkbox') {
      input = document.createElement('input');
      input.type = 'checkbox';
      input.id = 'param_' + inputIdSuffix;
      input.checked = param.default === true || param.default === 'true';
      input.style.marginRight = '8px';
      input.style.width = 'auto';
      
      input.onchange = function() {
        setStateParam(statePath, input.checked);
      };
      setStateParam(statePath, input.checked);
      
      // For checkbox, wrap label and input together
      var labelContainer = document.createElement('div');
      labelContainer.style.display = 'flex';
      labelContainer.style.alignItems = 'center';
      labelContainer.style.gap = '8px';
      
      labelContainer.appendChild(input);
      labelContainer.appendChild(label);
      group.appendChild(labelContainer);
      
      // Skip the normal label append below
      label = null;
      
    } else if (param.type === 'time_range') {
      // Custom time range input with checkbox
      var timeRangeContainer = document.createElement('div');
      timeRangeContainer.style.marginTop = '8px';
      
      // Checkbox to enable time range
      var checkboxContainer = document.createElement('div');
      checkboxContainer.style.display = 'flex';
      checkboxContainer.style.alignItems = 'center';
      checkboxContainer.style.gap = '8px';
      checkboxContainer.style.marginBottom = '12px';
      
      var enableCheckbox = document.createElement('input');
      enableCheckbox.type = 'checkbox';
      enableCheckbox.id = 'param_' + inputIdSuffix + '_enable';
      enableCheckbox.style.width = 'auto';
      enableCheckbox.style.marginRight = '8px';
      
      var checkboxLabel = document.createElement('label');
      checkboxLabel.textContent = 'Aktifkan Jadwal Operasional';
      checkboxLabel.style.cursor = 'pointer';
      checkboxLabel.onclick = function() { enableCheckbox.click(); };
      
      checkboxContainer.appendChild(enableCheckbox);
      checkboxContainer.appendChild(checkboxLabel);
      timeRangeContainer.appendChild(checkboxContainer);
      
      // Time inputs container (hidden by default)
      var timeInputsContainer = document.createElement('div');
      timeInputsContainer.id = 'param_' + inputIdSuffix + '_inputs';
      timeInputsContainer.style.display = 'none';
      timeInputsContainer.style.paddingLeft = '28px';
      timeInputsContainer.style.borderLeft = '3px solid var(--primary)';
      
      // Start time
      var startTimeDiv = document.createElement('div');
      startTimeDiv.style.marginBottom = '12px';
      var startLabel = document.createElement('label');
      startLabel.textContent = 'Jam Mulai';
      startLabel.style.display = 'block';
      startLabel.style.marginBottom = '6px';
      startLabel.style.fontSize = '13px';
      startTimeDiv.appendChild(startLabel);
      
      var startTimeContainer = document.createElement('div');
      startTimeContainer.style.display = 'flex';
      startTimeContainer.style.gap = '8px';
      startTimeContainer.style.alignItems = 'center';
      
      var startHour = createTimeInput('param_' + inputIdSuffix + '_start_hour', '07', '00', '23');
      var startMin = createTimeInput('param_' + inputIdSuffix + '_start_min', '00', '00', '59');
      var startSec = createTimeInput('param_' + inputIdSuffix + '_start_sec', '00', '00', '59');
      
      startTimeContainer.appendChild(startHour);
      startTimeContainer.appendChild(createTimeSeparator(':'));
      startTimeContainer.appendChild(startMin);
      startTimeContainer.appendChild(createTimeSeparator(':'));
      startTimeContainer.appendChild(startSec);
      startTimeDiv.appendChild(startTimeContainer);
      timeInputsContainer.appendChild(startTimeDiv);
      
      // End time
      var endTimeDiv = document.createElement('div');
      endTimeDiv.style.marginBottom = '8px';
      var endLabel = document.createElement('label');
      endLabel.textContent = 'Jam Selesai';
      endLabel.style.display = 'block';
      endLabel.style.marginBottom = '6px';
      endLabel.style.fontSize = '13px';
      endTimeDiv.appendChild(endLabel);
      
      var endTimeContainer = document.createElement('div');
      endTimeContainer.style.display = 'flex';
      endTimeContainer.style.gap = '8px';
      endTimeContainer.style.alignItems = 'center';
      
      var endHour = createTimeInput('param_' + inputIdSuffix + '_end_hour', '17', '00', '23');
      var endMin = createTimeInput('param_' + inputIdSuffix + '_end_min', '00', '00', '59');
      var endSec = createTimeInput('param_' + inputIdSuffix + '_end_sec', '00', '00', '59');
      
      endTimeContainer.appendChild(endHour);
      endTimeContainer.appendChild(createTimeSeparator(':'));
      endTimeContainer.appendChild(endMin);
      endTimeContainer.appendChild(createTimeSeparator(':'));
      endTimeContainer.appendChild(endSec);
      endTimeDiv.appendChild(endTimeContainer);
      timeInputsContainer.appendChild(endTimeDiv);
      
      timeRangeContainer.appendChild(timeInputsContainer);
      
      // Toggle visibility and update state
      enableCheckbox.onchange = function() {
        if (enableCheckbox.checked) {
          timeInputsContainer.style.display = 'block';
          updateTimeRangeValue();
        } else {
          timeInputsContainer.style.display = 'none';
          setStateParam(statePath, '');
        }
      };
      
      // Function to update time range value
      function updateTimeRangeValue() {
        if (!enableCheckbox.checked) {
          setStateParam(statePath, '');
          return;
        }
        var startTime = pad(startHour.value) + ':' + pad(startMin.value) + ':' + pad(startSec.value);
        var endTime = pad(endHour.value) + ':' + pad(endMin.value) + ':' + pad(endSec.value);
        setStateParam(statePath, startTime + '-' + endTime);
      }
      
      // Attach onchange to all time inputs
      [startHour, startMin, startSec, endHour, endMin, endSec].forEach(function(input) {
        input.onchange = updateTimeRangeValue;
        input.onkeyup = updateTimeRangeValue;
      });
      
      // Set default
      setStateParam(statePath, '');
      
      // Use timeRangeContainer as the input
      input = timeRangeContainer;
      
    } else {
      // Default: number or text input
      input = document.createElement('input');
      input.type = param.type;
      input.id = 'param_' + inputIdSuffix;
      input.value = param.default;
      input.onchange = function() {
        var parsed = parseFloat(input.value);
        setStateParam(statePath, Number.isNaN(parsed) ? input.value : parsed);
      };
      setStateParam(statePath, param.default);
    }
    
    if (label) {
      group.appendChild(label);
    }
    
    if (input && param.type !== 'checkbox') {
      group.appendChild(input);
    }
    
    // Add description if available
    if (param.description) {
      var desc = document.createElement('small');
      desc.style.display = 'block';
      desc.style.marginTop = '4px';
      desc.style.color = 'var(--muted)';
      desc.style.fontSize = '12px';
      desc.style.lineHeight = '1.4';
      desc.textContent = param.description;
      group.appendChild(desc);
    }
    
    container.appendChild(group);
  });
  
  paramsDiv.classList.remove('hidden');
}

function buildFeatureOverview(params, keys) {
  if (!keys || keys.length === 0) {
    return null;
  }

  var groups = {
    basic: [],
    tracking: [],
    advanced: [],
  };

  keys.forEach(function(key) {
    var param = params[key];
    if (!param) {
      return;
    }

    var groupKey = getParamGroup(key);
    groups[groupKey].push(param.label || key);
  });

  var wrapper = document.createElement('div');
  wrapper.className = 'feature-overview';

  var title = document.createElement('div');
  title.className = 'feature-overview-title';
  title.textContent = 'Ringkasan Parameter';
  wrapper.appendChild(title);

  var subtitle = document.createElement('div');
  subtitle.className = 'feature-overview-subtitle';
  subtitle.textContent = 'Daftar parameter aktif per grup untuk fitur ini';
  wrapper.appendChild(subtitle);

  appendFeatureOverviewGroup(wrapper, 'Basic', groups.basic);
  appendFeatureOverviewGroup(wrapper, 'Tracking', groups.tracking);
  appendFeatureOverviewGroup(wrapper, 'Advanced', groups.advanced);

  return wrapper;
}

function appendFeatureOverviewGroup(parent, groupName, items) {
  if (!items || items.length === 0) {
    return;
  }

  var group = document.createElement('div');
  group.className = 'feature-overview-group';

  var heading = document.createElement('div');
  heading.className = 'feature-overview-heading';
  heading.textContent = groupName + ' (' + items.length + ')';
  group.appendChild(heading);

  var chips = document.createElement('div');
  chips.className = 'feature-overview-chips';

  items.forEach(function(item) {
    var chip = document.createElement('span');
    chip.className = 'feature-overview-chip';
    chip.textContent = item;
    chips.appendChild(chip);
  });

  group.appendChild(chips);
  parent.appendChild(group);
}

function getParamGroup(key) {
  if (key.indexOf('tracking_') !== 0) {
    return 'basic';
  }

  if (
    key === 'tracking_bt_track_high_thresh' ||
    key === 'tracking_bt_track_low_thresh' ||
    key === 'tracking_bt_new_track_thresh' ||
    key === 'tracking_bt_match_thresh' ||
    key === 'tracking_reid_similarity_thresh' ||
    key === 'tracking_reid_momentum'
  ) {
    return 'advanced';
  }

  return 'tracking';
}

function setStateParam(path, value) {
  var parts = path.split('.');
  var ref = state.featureParams;
  for (var i = 0; i < parts.length - 1; i++) {
    if (!ref[parts[i]] || typeof ref[parts[i]] !== 'object') {
      ref[parts[i]] = {};
    }
    ref = ref[parts[i]];
  }
  ref[parts[parts.length - 1]] = value;
}

function getParamPriority(key) {
  if (key === 'time_allowed') return 999;
  if (key === 'tracking_tracker_impl') return 200;
  if (key === 'tracking_bt_track_buffer') return 210;
  if (key === 'tracking_bt_fuse_score') return 220;
  if (key === 'tracking_bt_track_high_thresh') return 230;
  if (key === 'tracking_bt_track_low_thresh') return 240;
  if (key === 'tracking_bt_new_track_thresh') return 250;
  if (key === 'tracking_bt_match_thresh') return 260;
  if (key === 'tracking_reid_enabled') return 270;
  if (key === 'tracking_reid_similarity_thresh') return 280;
  if (key === 'tracking_reid_momentum') return 290;
  if (key.indexOf('tracking_') === 0) return 300;
  return 100;
}

// Helper function to create time input
function createTimeInput(id, placeholder, min, max) {
  var input = document.createElement('input');
  input.type = 'number';
  input.id = id;
  input.value = placeholder;
  input.min = min;
  input.max = max;
  input.placeholder = placeholder;
  input.style.width = '60px';
  input.style.padding = '6px 8px';
  input.style.textAlign = 'center';
  input.style.fontSize = '14px';
  input.style.border = '1px solid var(--border)';
  input.style.borderRadius = '4px';
  input.style.background = 'var(--panel-2)';
  input.style.color = 'var(--text)';
  return input;
}

function createTimeSeparator(text) {
  var span = document.createElement('span');
  span.textContent = text;
  span.style.fontSize = '18px';
  span.style.fontWeight = 'bold';
  span.style.color = 'var(--muted)';
  return span;
}

function pad(num) {
  var n = parseInt(num) || 0;
  return n < 10 ? '0' + n : '' + n;
}

// ===== SOURCE MODE =====
function setupSourceMode() {
  document.getElementById('btnProduction').onclick = function() {
    setSourceMode('production');
  };
  
  document.getElementById('btnInference').onclick = function() {
    setSourceMode('inference');
  };
  
  setSourceMode('inference');
}

function setSourceMode(mode) {
  state.sourceMode = mode;
  
  document.getElementById('productionInputs').classList.toggle('hidden', mode !== 'production');
  document.getElementById('inferenceInputs').classList.toggle('hidden', mode !== 'inference');
  
  // Button styling
  document.getElementById('btnProduction').style.background = mode === 'production' ? '#1f4f86' : '#233047';
  document.getElementById('btnInference').style.background = mode === 'inference' ? '#1f4f86' : '#233047';
  
  log('SOURCE', 'Mode: ' + mode + (mode === 'production' ? ' (looping)' : ' (non-looping)'));
}



// ===== LOGS CONFIG =====
function setupLogsConfig() {
  var enableLogReading = document.getElementById('enableLogReading');
  var logReadingOptions = document.getElementById('logReadingOptions');
  
  enableLogReading.onchange = function() {
    if (enableLogReading.checked) {
      logReadingOptions.classList.remove('hidden');
      log('LOGS', 'Log reading enabled - re-inference mode activated');
    } else {
      logReadingOptions.classList.add('hidden');
      log('LOGS', 'Log reading disabled');
    }
  };
}

// ===== IMAGE UPLOAD =====
function setupImageUpload() {
  var imageFile = document.getElementById('imageFile');
  var btnClearImage = document.getElementById('btnClearImage');
  
  // Handle image selection
  imageFile.onchange = function(e) {
    if (e.target.files.length === 0) return;
    
    var file = e.target.files[0];
    var reader = new FileReader();
    
    reader.onload = function(event) {
      var img = new Image();
      
      img.onload = function() {
        // Store image
        state.backgroundImage = img;
        
        // Redraw canvas with image
        redrawCanvas();
        
        log('IMAGE', 'Image loaded: ' + file.name + ' (' + img.width + 'x' + img.height + ')');
        showSuccess('Image loaded successfully!');
      };
      
      img.onerror = function() {
        log('ERROR', 'Failed to load image: ' + file.name);
        showError('Failed to load image');
      };
      
      img.src = event.target.result;
    };
    
    reader.onerror = function() {
      log('ERROR', 'Failed to read file');
      showError('Failed to read file');
    };
    
    reader.readAsDataURL(file);
  };
  
  // Clear image
  btnClearImage.onclick = function() {
    state.backgroundImage = null;
    document.getElementById('imageFile').value = '';
    redrawCanvas();
    log('IMAGE', 'Image cleared');
    showSuccess('Image cleared');
  };
}

// ===== CANVAS SETUP =====
function setupCanvas() {
  canvas.onclick = function(e) {
    // If line direction picker is active, handle direction selection instead
    if (state.lineDirectionPickerActive) {
      handleLineDirectionSelection(e);
      return;
    }
    
    if (!state.drawMode) return;
    
    var rect = canvas.getBoundingClientRect();
    var x = (e.clientX - rect.left) / rect.width;
    var y = (e.clientY - rect.top) / rect.height;
    
    addPoint(x, y);
  };
}

function addPoint(x, y) {
  state.currentPoints.push({ x: x, y: y });
  
  log('DRAW', 'Point added: (' + x.toFixed(3) + ', ' + y.toFixed(3) + ')');
  
  // Enable save button based on requirements
  var canSave = false;
  if (state.drawMode === 'region' && state.currentPoints.length >= 3) {
    canSave = true;
  } else if (state.drawMode === 'line' && state.currentPoints.length === 2) {
    canSave = true;
  }
  
  document.getElementById('btnSaveGeometry').disabled = !canSave;
  
  redrawCanvas();
}

function redrawCanvas() {
  // Clear
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Draw background image if loaded
  if (state.backgroundImage) {
    try {
      // Draw image to fit canvas
      ctx.drawImage(state.backgroundImage, 0, 0, canvas.width, canvas.height);
    } catch (e) {
      log('ERROR', 'Failed to draw background image: ' + e.message);
    }
  }
  
  // Draw saved regions
  state.regions.forEach(function(region) {
    drawPolygon(region.coords, region.color, 0.2);
    drawPolygonOutline(region.coords, region.color);
  });
  
  // Draw saved lines
  state.lines.forEach(function(line) {
    drawLine(line.coords, line.color);
    // Draw centroid for line_cross feature
    if (line.centroid) {
      drawCentroid(line.centroid, line.color);
    }
    
    // Show IN/OUT visualization only after direction chosen
    if (line.direction && line.coords && line.coords.length === 2) {
      drawLineDirectionPicker({
        coords: line.coords,
        color: line.color,
        direction: line.direction,
        pickerStyle: 'result'
      });
    }
  });
  
  // ACTIVE PICKER: Draw NEUTRAL areas until user clicks
  if (state.lineDirectionPickerActive && state.lineDirectionPickerLine) {
    drawLineDirectionPicker({
      coords: state.lineDirectionPickerLine.coords,
      color: state.lineDirectionPickerLine.color,
      pickerStyle: 'neutral'
    });
  }
  
  // Draw current points (preview)
  if (state.currentPoints.length > 0) {
    if (state.drawMode === 'region') {
      drawPolygon(state.currentPoints, state.selectedColor, 0.3);
      drawPolygonOutline(state.currentPoints, state.selectedColor);
    } else if (state.drawMode === 'line') {
      if (state.currentPoints.length === 1) {
        drawPoint(state.currentPoints[0], state.selectedColor);
      } else if (state.currentPoints.length === 2) {
        drawLine(state.currentPoints, state.selectedColor);
      }
    }
  }
}

function drawPolygon(points, color, alpha) {
  if (points.length < 3) return;
  
  ctx.fillStyle = color + Math.floor(alpha * 255).toString(16).padStart(2, '0');
  ctx.beginPath();
  ctx.moveTo(points[0].x * canvas.width, points[0].y * canvas.height);
  for (var i = 1; i < points.length; i++) {
    ctx.lineTo(points[i].x * canvas.width, points[i].y * canvas.height);
  }
  ctx.closePath();
  ctx.fill();
}

function drawPolygonOutline(points, color) {
  if (points.length < 2) return;
  
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(points[0].x * canvas.width, points[0].y * canvas.height);
  for (var i = 1; i < points.length; i++) {
    ctx.lineTo(points[i].x * canvas.width, points[i].y * canvas.height);
  }
  if (points.length >= 3) {
    ctx.closePath();
  }
  ctx.stroke();
  
  // Draw points
  points.forEach(function(p) {
    drawPoint(p, color);
  });
}

function drawLine(points, color) {
  if (points.length !== 2) return;
  
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(points[0].x * canvas.width, points[0].y * canvas.height);
  ctx.lineTo(points[1].x * canvas.width, points[1].y * canvas.height);
  ctx.stroke();
  
  // Draw points
  points.forEach(function(p) {
    drawPoint(p, color);
  });
}

function drawPoint(point, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(point.x * canvas.width, point.y * canvas.height, 5, 0, Math.PI * 2);
  ctx.fill();
}

function drawCentroid(point, color) {
  // Draw centroid as a larger circle with a crosshair
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(point.x * canvas.width, point.y * canvas.height, 8, 0, Math.PI * 2);
  ctx.stroke();
  
  // Draw crosshair
  var x = point.x * canvas.width;
  var y = point.y * canvas.height;
  ctx.beginPath();
  ctx.moveTo(x - 6, y);
  ctx.lineTo(x + 6, y);
  ctx.moveTo(x, y - 6);
  ctx.lineTo(x, y + 6);
  ctx.stroke();
}

function drawLineDirectionPicker(lineData) {
  var pickerStyle = lineData.pickerStyle || 'neutral';
  var p1 = lineData.coords[0];
  var p2 = lineData.coords[1];
  var color = lineData.color || '#0080ff';

  // Convert to canvas coords
  var x1 = p1.x * canvas.width;
  var y1 = p1.y * canvas.height;
  var x2 = p2.x * canvas.width;
  var y2 = p2.y * canvas.height;

  // Calculate perpendicular direction (canvas coords)
  var dx = x2 - x1;
  var dy = y2 - y1;
  var len = Math.sqrt(dx * dx + dy * dy) || 1;
  var perpX = -dy / len;
  var perpY = dx / len;

  var offset = 120;

  // Decide which side is IN in result mode
  var inIsPositivePerp = true;
  if (pickerStyle === 'result' && lineData.direction && lineData.direction !== 'bidirectional') {
    var dirVec = { x: 0, y: 0 };
    if (lineData.direction === 'upward') dirVec = { x: 0, y: -1 };
    else if (lineData.direction === 'downward') dirVec = { x: 0, y: 1 };
    else if (lineData.direction === 'leftward') dirVec = { x: -1, y: 0 };
    else if (lineData.direction === 'rightward') dirVec = { x: 1, y: 0 };

    var dot = dirVec.x * perpX + dirVec.y * perpY;
    inIsPositivePerp = dot >= 0;
  }

  // Colors - LIGHTER grays for neutral to be more visible
  var neutralFill = 'rgba(200,200,200,0.35)';  // LIGHTER neutral
  var neutralStroke = 'rgba(150,150,150,0.95)';  // Darker border for contrast
  var inFill = 'rgba(0,200,0,0.50)';  // Bright green for IN
  var inStroke = 'rgba(0,150,0,1.0)';
  var outFill = 'rgba(255,80,80,0.50)';  // Bright red for OUT
  var outStroke = 'rgba(200,0,0,1.0)';

  var positiveFill = pickerStyle === 'neutral' ? neutralFill : (inIsPositivePerp ? inFill : outFill);
  var positiveStroke = pickerStyle === 'neutral' ? neutralStroke : (inIsPositivePerp ? inStroke : outStroke);
  var negativeFill = pickerStyle === 'neutral' ? neutralFill : (inIsPositivePerp ? outFill : inFill);
  var negativeStroke = pickerStyle === 'neutral' ? neutralStroke : (inIsPositivePerp ? outStroke : inStroke);

  // Negative side trapezoid
  ctx.globalAlpha = 1.0;
  ctx.fillStyle = negativeFill;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - perpX * offset, y1 - perpY * offset);
  ctx.lineTo(x2 - perpX * offset, y2 - perpY * offset);
  ctx.lineTo(x2, y2);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = negativeStroke;
  ctx.lineWidth = 6;
  ctx.beginPath();
  ctx.moveTo(x1 - perpX * offset, y1 - perpY * offset);
  ctx.lineTo(x2 - perpX * offset, y2 - perpY * offset);
  ctx.stroke();

  // Positive side trapezoid
  ctx.fillStyle = positiveFill;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 + perpX * offset, y1 + perpY * offset);
  ctx.lineTo(x2 + perpX * offset, y2 + perpY * offset);
  ctx.lineTo(x2, y2);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = positiveStroke;
  ctx.lineWidth = 6;
  ctx.beginPath();
  ctx.moveTo(x1 + perpX * offset, y1 + perpY * offset);
  ctx.lineTo(x2 + perpX * offset, y2 + perpY * offset);
  ctx.stroke();

  // Draw main line (EXTRA THICK for visibility)
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 8;  // EXTRA THICK
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();

  // Draw main line with original color
  ctx.strokeStyle = color;
  ctx.lineWidth = 5;  // INCREASED from 2
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();

  // Draw points
  drawPoint(p1, color);
  drawPoint(p2, color);

  // Draw LARGE centroid - SUPER VISIBLE
  var centroid = {
    x: (p1.x + p2.x) / 2,
    y: (p1.y + p2.y) / 2
  };

  var centroidX = centroid.x * canvas.width;
  var centroidY = centroid.y * canvas.height;

  // Large yellow circle as backdrop
  ctx.fillStyle = '#ffff00';
  ctx.globalAlpha = 0.7;
  ctx.beginPath();
  ctx.arc(centroidX, centroidY, 16, 0, Math.PI * 2);
  ctx.fill();

  // White circle outline
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 3;
  ctx.globalAlpha = 1.0;
  ctx.beginPath();
  ctx.arc(centroidX, centroidY, 16, 0, Math.PI * 2);
  ctx.stroke();

  // Thick crosshair
  ctx.strokeStyle = '#000000';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(centroidX - 10, centroidY);
  ctx.lineTo(centroidX + 10, centroidY);
  ctx.moveTo(centroidX, centroidY - 10);
  ctx.lineTo(centroidX, centroidY + 10);
  ctx.stroke();

  // Draw labels with LARGER font and strong shadow
  ctx.font = 'bold 40px Arial';  // EVEN LARGER for clarity
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.shadowColor = 'rgba(0,0,0,0.8)';
  ctx.shadowBlur = 8;
  ctx.shadowOffsetX = 2;
  ctx.shadowOffsetY = 2;

  var negX = (x1 + x2) / 2 - perpX * (offset / 2);
  var negY = (y1 + y2) / 2 - perpY * (offset / 2);
  var posX = (x1 + x2) / 2 + perpX * (offset / 2);
  var posY = (y1 + y2) / 2 + perpY * (offset / 2);

  if (pickerStyle === 'neutral') {
    // NEUTRAL mode - show "NETRAL" with clear labels
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = '#1a1a1a';  // Dark text
    ctx.fillText('NETRAL', negX, negY - 20);
    ctx.font = 'bold 32px Arial';
    ctx.fillText('← KLIK SINI', negX, negY + 25);
    
    ctx.font = 'bold 40px Arial';
    ctx.fillStyle = '#1a1a1a';
    ctx.fillText('NETRAL', posX, posY - 20);
    ctx.font = 'bold 32px Arial';
    ctx.fillText('KLIK SINI →', posX, posY + 25);
  } else {
    // RESULT mode - show IN/OUT with colors
    var negIsIn = !inIsPositivePerp;
    var posIsIn = inIsPositivePerp;
    
    // Draw clear labels with directional arrows (8-way support)
    // 8-direction arrows follow perpendicular angle precisely:
    // ↑↗→↘↓↙←↖ (N, NE, E, SE, S, SW, W, NW)
    // CLOCK ANALOGY: 
    //   - Line 9→3 (horizontal): arrows 12↑ & 6↓
    //   - Line 11→5 (diagonal): arrows 2↗ & 8↙ (MENGIKUTI KEMIRINGAN!)
    var negArrow = '';
    var posArrow = '';
    
    // Calculate perpendicular angle (in degrees, 0° = right, 90° = up)
    var perpAngle = Math.atan2(-perpY, perpX) * 180 / Math.PI;  // -perpY because canvas Y is inverted
    
    // Determine arrow from perpendicular angle (8 directions)
    // Positive side points in perpendicular direction
    // Negative side points opposite (-180°)
    function angleToArrow(angleDeg) {
      // Normalize to 0-360
      var a = ((angleDeg % 360) + 360) % 360;
      
      // 8 directions with 45° ranges
      if (a >= 337.5 || a < 22.5) return '→';      // East (0°)
      else if (a >= 22.5 && a < 67.5) return '↗';  // Northeast (45°)
      else if (a >= 67.5 && a < 112.5) return '↑'; // North (90°)
      else if (a >= 112.5 && a < 157.5) return '↖'; // Northwest (135°)
      else if (a >= 157.5 && a < 202.5) return '←'; // West (180°)
      else if (a >= 202.5 && a < 247.5) return '↙'; // Southwest (225°)
      else if (a >= 247.5 && a < 292.5) return '↓'; // South (270°)
      else return '↘';  // Southeast (315°)
    }
    
    posArrow = angleToArrow(perpAngle);           // Points in perpendicular direction
    negArrow = angleToArrow(perpAngle + 180);     // Points opposite direction
    
    ctx.font = 'bold 40px Arial';
    ctx.globalAlpha = 1.0;
    
    // Negative side label
    ctx.fillStyle = negIsIn ? '#00dd00' : '#ff3333';
    ctx.fillText(negArrow + ' ' + (negIsIn ? 'IN' : 'OUT'), negX, negY);
    
    // Positive side label
    ctx.fillStyle = posIsIn ? '#00dd00' : '#ff3333';
    ctx.fillText((posIsIn ? 'IN' : 'OUT') + ' ' + posArrow, posX, posY);
  }

  ctx.shadowColor = 'transparent';
  ctx.globalAlpha = 1.0;
}

// ===== LINE DIRECTION DETECTION =====
function detectLineDirection(p1, p2) {
  /**
   * Deteksi direction dari 2 points (line)
   * Returns: { axis, angle, slope }
   * axis: 'horizontal', 'vertical', atau 'diagonal'
   * angle: sudut dalam derajat (0-360)
   * slope: kemiringan (rise/run)
   */
  var dx = p2.x - p1.x;
  var dy = p2.y - p1.y;
  
  // Calculate angle in degrees (0-360)
  var angleRad = Math.atan2(dy, dx);
  var angleDeg = (angleRad * 180 / Math.PI + 360) % 360;
  
  // Normalize to 0-180 (line doesn't have direction, so >180 equals <180 from opposite)
  if (angleDeg > 180) {
    angleDeg = 360 - angleDeg;
  }
  
  // Detect axis
  var axis = 'diagonal';
  if (Math.abs(dx) < 0.01) {
    axis = 'vertical';
  } else if (Math.abs(dy) < 0.01) {
    axis = 'horizontal';
  } else if (angleDeg >= 40 && angleDeg <= 50) {
    axis = 'diagonal';
  }
  
  return {
    axis: axis,
    angle: angleDeg,
    slope: Math.abs(dy) / (Math.abs(dx) || 0.001),
    dx: dx,
    dy: dy
  };
}

function getOppositeDirection(direction) {
  var opposites = {
    'left-to-right': 'right-to-left',
    'right-to-left': 'left-to-right',
    'top-to-bottom': 'bottom-to-top',
    'bottom-to-top': 'top-to-bottom'
  };
  return opposites[direction] || direction;
}

// ===== LINE DIRECTION PICKER =====
function startLineDirectionPicker(lineData) {
  /**
   * Start interactive direction picker for line_cross
  * Neutral preview first; user clicks one side to set IN.
   */
  console.log('[LINE-PICKER] ★ STARTED - coords:', lineData.coords, 'color:', lineData.color);
  
  state.lineDirectionPickerActive = true;
  state.lineDirectionPickerLine = lineData;
  state.drawMode = null;  // Exit drawing mode
  state.currentPoints = [];
  
  // Show instructions
  document.getElementById('directionPickerInstructions').style.display = 'block';
  
  log('PICKER', 'Line direction picker started - neutral areas, click one side = IN');
  showStatus('Line drawn. Area masih netral. Klik salah satu sisi untuk set IN. Sisi lain otomatis OUT.', 'info');
  
  console.log('[LINE-PICKER] Picker state active:', state.lineDirectionPickerActive);
  console.log('[LINE-PICKER] Calling redrawCanvas next...');
  
  redrawCanvas();
}

function handleLineDirectionSelection(mouseEvent) {
  /**
   * Line Direction Picker - Click ONE side = that side is IN, other side = OUT
   * Support for direction modes: leftward, rightward, upward, downward, bidirectional
   */
  if (!state.lineDirectionPickerActive || !state.lineDirectionPickerLine) {
    return;
  }
  
  var rect = canvas.getBoundingClientRect();
  var mouseX = mouseEvent.clientX - rect.left;
  var mouseY = mouseEvent.clientY - rect.top;
  
  // Normalize to 0-1
  var normX = mouseX / canvas.width;
  var normY = mouseY / canvas.height;

  var line = state.lineDirectionPickerLine;
  var p1 = line.coords[0];
  var p2 = line.coords[1];
  
  // Calculate perpendicular direction
  var dx = p2.x - p1.x;
  var dy = p2.y - p1.y;
  var len = Math.sqrt(dx * dx + dy * dy) || 1;
  var perpX = -dy / len;
  var perpY = dx / len;
  
  // Check which side user clicked
  var clickVecX = normX - (p1.x + p2.x) / 2;
  var clickVecY = normY - (p1.y + p2.y) / 2;
  var dotProduct = clickVecX * perpX + clickVecY * perpY;
  
  var clickedNegativeSide = dotProduct < 0;  // negative = left/up side

  // Determine IN normal vector in normalized space
  var inNormalX = clickedNegativeSide ? -perpX : perpX;
  var inNormalY = clickedNegativeSide ? -perpY : perpY;

  // Calculate centroid
  var centroid = {
    x: (p1.x + p2.x) / 2,
    y: (p1.y + p2.y) / 2
  };
  
  // Get direction detection
  var detection = detectLineDirection(p1, p2);
  var orientation = detection.axis;
  
  // UNIVERSAL DIRECTION MAPPING - works for ANY line angle!
  // Direction = arah IN (tegak lurus ke garis)
  // Logic: pilih direction cardinal (4 arah) yang paling dekat dengan IN normal vector
  // Note: canvas Y grows downward (Y+ = bawah, Y- = atas)
  var simpleDirection;
  
  // Selalu gunakan component terbesar dari IN normal vector
  if (Math.abs(inNormalX) >= Math.abs(inNormalY)) {
    // Component X lebih besar → direction horizontal (leftward/rightward)
    simpleDirection = inNormalX >= 0 ? 'rightward' : 'leftward';
  } else {
    // Component Y lebih besar → direction vertical (upward/downward)
    simpleDirection = inNormalY >= 0 ? 'downward' : 'upward';
  }
  
  // Create final line with COMPLETE properties
  var lineId = state.lineIdCounter++;
  var finalLine = {
    id: lineId,
    type: 'line_' + lineId,
    name: 'Line #' + lineId,
    color: line.color,
    coords: line.coords,
    centroid: centroid,
    direction: simpleDirection,      // leftward, rightward, upward, downward, or bidirectional
    orientation: orientation,         // horizontal, vertical, or diagonal
    bidirectional: false,             // Set to false by default, can be modified later
    mode: []
  };
  
  state.lines.push(finalLine);
  updateLineList();
  
  // Reset picker
  state.lineDirectionPickerActive = false;
  state.lineDirectionPickerLine = null;
  
  // Hide instructions
  document.getElementById('directionPickerInstructions').style.display = 'none';
  
  document.getElementById('btnCancelDrawing').disabled = true;
  document.getElementById('btnSaveGeometry').disabled = true;
  
  log('SAVE', 'Line #' + finalLine.id + ' saved | direction: ' + simpleDirection);
  showSuccess('Line saved. Direction: ' + simpleDirection);
  
  redrawCanvas();
}

function cancelLineDirectionPicker() {
  state.lineDirectionPickerActive = false;
  state.lineDirectionPickerLine = null;
  state.currentPoints = [];
  state.drawMode = null;
  
  // Hide instructions
  document.getElementById('directionPickerInstructions').style.display = 'none';
  
  document.getElementById('btnCancelDrawing').disabled = true;
  document.getElementById('btnSaveGeometry').disabled = true;
  log('CANCEL', 'Line direction picker cancelled');
  redrawCanvas();
}

// ===== DRAWING CONTROLS =====
function setupDrawingControls() {
  document.getElementById('btnCreateRegion').onclick = function() {
    startDrawing('region');
  };
  
  document.getElementById('btnCreateLine').onclick = function() {
    startDrawing('line');
  };
  
  document.getElementById('btnSaveGeometry').onclick = function() {
    saveGeometry();
  };
  
  document.getElementById('btnCancelDrawing').onclick = function() {
    cancelDrawing();
  };
  
  document.getElementById('btnClearAll').onclick = function() {
    clearAll();
  };
}

function startDrawing(mode) {
  state.drawMode = mode;
  state.currentPoints = [];
  
  document.getElementById('btnCancelDrawing').disabled = false;
  document.getElementById('btnSaveGeometry').disabled = true;
  
  log('DRAW', 'Started drawing ' + mode);
}

function saveGeometry() {
  if (!state.drawMode || state.currentPoints.length === 0) return;
  
  if (state.drawMode === 'region') {
    if (state.currentPoints.length < 3) {
      showError('Region must have at least 3 points');
      return;
    }
    
    var regionId = state.regionIdCounter++;
    var region = {
      id: regionId,
      type: 'region_' + (regionId),
      name: 'Region #' + regionId,
      color: state.selectedColor,
      coords: state.currentPoints.slice(),
      mode: []
    };
    
    state.regions.push(region);
    updateRegionList();
    
    log('SAVE', 'Region #' + region.id + ' saved with ' + region.coords.length + ' points');
  } else if (state.drawMode === 'line') {
    if (state.currentPoints.length !== 2) {
      showError('Line must have exactly 2 points');
      return;
    }
    
    if (state.feature === 'line_cross') {
      // For line_cross: show direction picker so user can select in/out visually
      startLineDirectionPicker({
        coords: state.currentPoints.slice(),
        color: state.selectedColor
      });
      return;  // Don't save yet - wait for user to pick direction
    }
    
    // For other features: create line without centroid/direction
    var lineId = state.lineIdCounter++;
    var line = {
      id: lineId,
      type: 'line_' + lineId,
      name: 'Line #' + lineId,
      color: state.selectedColor,
      coords: state.currentPoints.slice(),
      mode: []
    };
    
    state.lines.push(line);
    updateLineList();
    
    log('SAVE', 'Line #' + line.id + ' saved');
  }
  
  // Reset
  state.drawMode = null;
  state.currentPoints = [];
  document.getElementById('btnCancelDrawing').disabled = true;
  document.getElementById('btnSaveGeometry').disabled = true;
  
  redrawCanvas();
}

function cancelDrawing() {
  // If line direction picker is active, cancel that instead
  if (state.lineDirectionPickerActive) {
    cancelLineDirectionPicker();
    return;
  }
  
  state.drawMode = null;
  state.currentPoints = [];
  
  document.getElementById('btnCancelDrawing').disabled = true;
  document.getElementById('btnSaveGeometry').disabled = true;
  
  log('DRAW', 'Drawing cancelled');
  redrawCanvas();
}

function clearAll() {
  if (!confirm('Clear all regions and lines?')) return;
  
  state.regions = [];
  state.lines = [];
  state.regionIdCounter = 1;
  state.lineIdCounter = 1;
  
  updateRegionList();
  updateLineList();
  redrawCanvas();
  
  log('CLEAR', 'All geometry cleared');
}

function deleteRegion(id) {
  state.regions = state.regions.filter(function(r) { return r.id !== id; });
  updateRegionList();
  redrawCanvas();
  log('DELETE', 'Region #' + id + ' deleted');
}

function deleteLine(id) {
  state.lines = state.lines.filter(function(l) { return l.id !== id; });
  updateLineList();
  redrawCanvas();
  log('DELETE', 'Line #' + id + ' deleted');
}

function updateRegionList() {
  var list = document.getElementById('regionList');
  list.innerHTML = '';
  
  state.regions.forEach(function(region) {
    var item = document.createElement('div');
    item.className = 'geometry-item';
    
    var info = document.createElement('div');
    var name = document.createElement('div');
    name.className = 'name';
    // Ensure name is properly set, fix old data if needed
    if (!region.name || region.name.indexOf('-') > 0 || region.name === region.type) {
      region.name = 'Region #' + region.id;
    }
    name.textContent = region.name;
    name.style.color = region.color;
    name.style.cursor = 'pointer';
    name.title = 'Double-click to edit name';
    
    // Double-click to edit name
    name.ondblclick = function() {
      var currentName = region.name || ('Region #' + region.id);
      var newName = prompt('Edit Region Name:', currentName);
      if (newName && newName.trim() !== '') {
        region.name = newName.trim();
        updateRegionList();
        redrawCanvas();
        log('EDIT', 'Region #' + region.id + ' renamed to "' + region.name + '"');
      }
    };
    
    var detail = document.createElement('div');
    detail.className = 'info';
    detail.textContent = region.coords.length + ' points';
    
    info.appendChild(name);
    info.appendChild(detail);
    
    var btn = document.createElement('button');
    btn.textContent = 'Delete';
    btn.className = 'danger';
    btn.onclick = function() {
      var regionName = region.name || ('Region #' + region.id);
      if (confirm('Delete ' + regionName + '?')) {
        deleteRegion(region.id);
      }
    };
    
    item.appendChild(info);
    item.appendChild(btn);
    list.appendChild(item);
  });
  
  document.getElementById('regionCount').textContent = state.regions.length;
}

function updateLineList() {
  var list = document.getElementById('lineList');
  list.innerHTML = '';
  
  state.lines.forEach(function(line) {
    var item = document.createElement('div');
    item.className = 'geometry-item';
    item.style.cursor = state.feature === 'line_cross' ? 'pointer' : 'default';
    
    var info = document.createElement('div');
    var name = document.createElement('div');
    name.className = 'name';
    // Ensure name is properly set, fix old data if needed
    if (!line.name || line.name.indexOf('-') > 0 || line.name === line.type) {
      line.name = 'Line #' + line.id;
    }
    name.textContent = line.name;
    name.style.color = line.color;
    name.style.cursor = 'pointer';
    name.title = 'Double-click to edit name';
    
    // Double-click to edit name
    name.ondblclick = function() {
      var currentName = line.name || ('Line #' + line.id);
      var newName = prompt('Edit Line Name:', currentName);
      if (newName && newName.trim() !== '') {
        line.name = newName.trim();
        updateLineList();
        redrawCanvas();
        log('EDIT', 'Line #' + line.id + ' renamed to "' + line.name + '"');
      }
    };
    
    var detail = document.createElement('div');
    detail.className = 'info';
    var detailText = '2 points';
    
    // For line_cross: keep it simple (no centroid X/Y numbers)
    if (line.direction) detailText += ' | Direction: ' + line.direction;
    if (line.orientation) detailText += ' | Orientation: ' + line.orientation;
    if (typeof line.bidirectional === 'boolean') {
      detailText += ' | Bidirectional: ' + (line.bidirectional ? 'true' : 'false');
    }
    
    detail.textContent = detailText;
    
    info.appendChild(name);
    info.appendChild(detail);
    
    // Make clickable to edit if line_cross feature is selected
    if (state.feature === 'line_cross') {
      info.style.cursor = 'pointer';
      info.onclick = function(e) {
        // Don't trigger edit when double-clicking name
        if (e.target === name) return;
        editLineProperties(line);
      };
    }
    
    var btn = document.createElement('button');
    btn.textContent = 'Delete';
    btn.className = 'danger';
    btn.onclick = function() {
      var lineName = line.name || ('Line #' + line.id);
      if (confirm('Delete ' + lineName + '?')) {
        deleteLine(line.id);
      }
    };
    
    item.appendChild(info);
    item.appendChild(btn);
    list.appendChild(item);
  });
  
  document.getElementById('lineCount').textContent = state.lines.length;
}

// ===== LINE PROPERTIES EDITOR (line_cross) =====
state.editingLineId = null;

function editLineProperties(line) {
  var editor = document.getElementById('linePropertiesEditor');
  if (!editor) return;

  state.editingLineId = line.id;
  document.getElementById('editingLineId').textContent = 'Line #' + line.id;

  var orientationEl = document.getElementById('lineOrientation');
  var directionEl = document.getElementById('lineDirection');
  var directionTextEl = document.getElementById('lineDirectionText');
  var bidirEl = document.getElementById('lineBidirectional');

  if (orientationEl) orientationEl.value = line.orientation || 'diagonal';

  var isPreset = (line.direction === 'upward' || line.direction === 'downward' || line.direction === 'leftward' || line.direction === 'rightward');
  if (directionEl) directionEl.value = isPreset ? line.direction : 'custom';
  if (directionTextEl) {
    directionTextEl.value = (!isPreset && line.direction) ? line.direction : '';
    directionTextEl.style.display = (directionEl && directionEl.value === 'custom') ? 'block' : 'none';
  }
  if (bidirEl) bidirEl.checked = !!line.bidirectional;

  if (directionEl && directionTextEl) {
    directionEl.onchange = function() {
      directionTextEl.style.display = (directionEl.value === 'custom') ? 'block' : 'none';
    };
  }

  editor.style.display = 'block';
}

function closeLinePropertiesEditor() {
  var editor = document.getElementById('linePropertiesEditor');
  if (!editor) return;
  editor.style.display = 'none';
  state.editingLineId = null;
}

function saveLineProperties() {
  if (!state.editingLineId) return;

  var line = null;
  for (var i = 0; i < state.lines.length; i++) {
    if (state.lines[i].id === state.editingLineId) {
      line = state.lines[i];
      break;
    }
  }
  if (!line) return;

  var orientationEl = document.getElementById('lineOrientation');
  var directionEl = document.getElementById('lineDirection');
  var directionTextEl = document.getElementById('lineDirectionText');
  var bidirEl = document.getElementById('lineBidirectional');

  if (orientationEl) line.orientation = orientationEl.value;

  var dir = directionEl ? directionEl.value : '';
  if (dir === 'custom') dir = directionTextEl ? directionTextEl.value.trim() : '';
  if (dir) line.direction = dir;

  if (bidirEl) line.bidirectional = !!bidirEl.checked;

  updateLineList();
  redrawCanvas();
  closeLinePropertiesEditor();
}

// ===== LOAD CONFIG =====
function setupLoadConfig() {
  document.getElementById('btnLoadConfig').onclick = function() {
    loadConfig();
  };
}

function loadConfig() {
  var fileInput = document.getElementById('configFile');
  var file = fileInput.files[0];
  
  if (!file) {
    showError('Please select a YAML file first');
    return;
  }
  
  log('LOAD', 'Loading config from: ' + file.name);
  
  var reader = new FileReader();
  reader.onload = function(e) {
    try {
      var yamlText = e.target.result;
      
      // Send to backend to parse and validate
      fetch('/api/parse', {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: yamlText
      })
      .then(function(res) {
        if (!res.ok) {
          return res.json().then(function(err) {
            throw new Error(err.detail || 'Parse failed');
          });
        }
        return res.json();
      })
      .then(function(config) {
        populateFromConfig(config);
        showSuccess('Config loaded successfully: ' + file.name);
        log('LOAD', 'Config loaded and populated');
      })
      .catch(function(err) {
        showError('Load failed: ' + err.message);
        log('ERROR', 'Load failed: ' + err.message);
      });
      
    } catch (err) {
      showError('Error reading file: ' + err.message);
      log('ERROR', 'File read error: ' + err.message);
    }
  };
  
  reader.onerror = function() {
    showError('Failed to read file');
    log('ERROR', 'FileReader error');
  };
  
  reader.readAsText(file);
}

function populateFromConfig(config) {
  // Config name
  var configName = config.metadata && config.metadata.config_name 
    ? config.metadata.config_name 
    : 'loaded-config';
  document.getElementById('configName').value = configName;
  
  // Model config
  if (config.model) {
    document.getElementById('modelName').value = config.model.name || 'person_office_detection';
    document.getElementById('device').value = config.model.device || 'cpu';
    document.getElementById('conf').value = config.model.conf || 0.25;
    document.getElementById('iou').value = config.model.iou || 0.7;
    document.getElementById('yoloType').value = config.model.type || 'small';
    
    if (config.model.classes && config.model.classes.length > 0) {
      document.getElementById('classes').value = config.model.classes.join(', ');
    }
  }
  
  // Feature selection
  if (config.feature && config.feature.length > 0) {
    var featureName = config.feature[0];
    // Map from YAML names to UI names
    if (featureName === 'dwelltime') featureName = 'dwell_time';
    if (featureName === 'linecross') featureName = 'line_cross';
    if (featureName === 'regioncrowd') featureName = 'region_crowd';
    
    selectFeature(featureName);
    
    // Feature params
    if (config.feature_params && config.feature_params[config.feature[0]]) {
      var params = config.feature_params[config.feature[0]];
      function applyParamsToInputs(obj, prefix) {
        Object.keys(obj).forEach(function(key) {
          var value = obj[key];
          var path = prefix ? (prefix + '.' + key) : key;

          if (value && typeof value === 'object' && !Array.isArray(value)) {
            applyParamsToInputs(value, path);
            return;
          }

          var inputId = 'param_' + path.replace(/\./g, '__');
          var input = document.getElementById(inputId);
          if (!input) {
            return;
          }

          if (input.type === 'checkbox') {
            input.checked = !!value;
          } else {
            input.value = value;
          }
          setStateParam(path, value);
        });
      }

      applyParamsToInputs(params, '');
    }
  }
  
  // Source input
  if (config.input) {
    if (config.input.rtsp_url) {
      setSourceMode('production');
      document.getElementById('rtspUrl').value = config.input.rtsp_url;
      document.getElementById('maxFpsProduction').value = config.input.max_fps || 30;
      
      // Load time_allowed if exists
      if (config.input.time_allowed) {
        if (config.input.time_allowed.start_time) {
          document.getElementById('timeAllowedStart').value = config.input.time_allowed.start_time;
        }
        if (config.input.time_allowed.end_time) {
          document.getElementById('timeAllowedEnd').value = config.input.time_allowed.end_time;
        }
      }
    } else if (config.input.video_files || config.input.video_source) {
      setSourceMode('inference');
      var loadedVideoSource = '';
      if (Array.isArray(config.input.video_files) && config.input.video_files.length > 0) {
        loadedVideoSource = config.input.video_files.join(', ');
      } else if (config.input.video_source) {
        loadedVideoSource = config.input.video_source;
      }
      document.getElementById('videoSource').value = loadedVideoSource;
      document.getElementById('maxFpsInference').value = config.input.max_fps || 30;
    }
  }
  
  // Logs configuration
  if (config.logs) {
    document.getElementById('logsBaseDir').value = config.logs.base_dir || 'logs';
    
    if (config.logs.enable_log_reading) {
      document.getElementById('enableLogReading').checked = true;
      document.getElementById('logReadingOptions').classList.remove('hidden');
      
      if (config.logs.enable_log_deletion) {
        document.getElementById('enableLogDeletion').checked = true;
      }
      
      if (config.logs.start_date) {
        document.getElementById('startDate').value = config.logs.start_date;
      }
      
      if (config.logs.end_date) {
        document.getElementById('endDate').value = config.logs.end_date;
      }
    }
  }
  
  // Regions
  state.regions = [];
  state.regionIdCounter = 1;
  if (config.regions && config.regions.length > 0) {
    config.regions.forEach(function(region) {
      state.regions.push({
        id: state.regionIdCounter++,
        type: region.type || 'region_' + state.regionIdCounter,
        name: region.name || ('Region #' + state.regionIdCounter),
        color: region.color || '#0080ff',
        coords: region.coords || [],
        mode: region.mode || []
      });
    });
    updateRegionList();
  }
  
  // Lines
  state.lines = [];
  state.lineIdCounter = 1;
  if (config.lines && config.lines.length > 0) {
    config.lines.forEach(function(line) {
      var newLine = {
        id: state.lineIdCounter++,
        type: line.type || 'line_' + state.lineIdCounter,
        name: line.name || ('Line #' + state.lineIdCounter),
        color: line.color || '#0080ff',
        coords: line.coords || [],
        mode: line.mode || []
      };
      
      // Restore centroid and direction for line_cross feature
      if (line.centroid) {
        newLine.centroid = line.centroid;
      }
      if (line.direction) {
        newLine.direction = line.direction;
      }
      if (line.orientation) {
        newLine.orientation = line.orientation;
      }
      if (line.bidirectional !== undefined) {
        newLine.bidirectional = line.bidirectional;
      }
      
      state.lines.push(newLine);
    });
    updateLineList();
  }
  
  // Redraw canvas with loaded geometry
  redrawCanvas();
  
  log('POPULATE', 'All fields populated from config');
}

// ===== BUILD & SAVE =====
function setupBuildControls() {
  document.getElementById('btnBuildPreview').onclick = function() {
    buildPreview();
  };
  
  document.getElementById('btnSaveConfig').onclick = function() {
    saveConfig();
  };
  
  document.getElementById('btnClearLogs').onclick = function() {
    clearLogs();
  };
  
  // Line properties editor handlers
  document.getElementById('btnSaveLineProps').onclick = function() {
    saveLineProperties();
  };
  
  document.getElementById('btnCancelLineProps').onclick = function() {
    cancelLineProperties();
  };
}

function buildPreview() {
  log('BUILD', 'Building YAML preview...');
  
  var request = collectFormData();
  
  fetch('/api/build', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  })
  .then(function(res) {
    if (!res.ok) {
      return res.json().then(function(err) {
        throw new Error(err.detail || 'Build failed');
      });
    }
    return res.text();
  })
  .then(function(yaml) {
    document.getElementById('yamlPreview').textContent = yaml;
    showSuccess('YAML preview built successfully');
    log('BUILD', 'Preview built successfully');
    loadLogs();
  })
  .catch(function(err) {
    showError('Build failed: ' + err.message);
    log('ERROR', 'Build failed: ' + err.message);
  });
}

function saveConfig() {
  log('SAVE', 'Saving configuration...');
  
  var request = collectFormData();
  
  fetch('/api/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  })
  .then(function(res) {
    if (!res.ok) {
      return res.json().then(function(err) {
        throw new Error(err.detail || 'Save failed');
      });
    }
    return res.json();
  })
  .then(function(data) {
    showSuccess('Config saved to: configs/app/' + data.filename);
    log('SAVE', 'Config saved: ' + data.filename + ' at ' + data.path);
    loadLogs();
  })
  .catch(function(err) {
    showError('Save failed: ' + err.message);
    log('ERROR', 'Save failed: ' + err.message);
  });
}

function collectFormData() {
  var configName = document.getElementById('configName').value || 'untitled';
  
  var model = {
    name: document.getElementById('modelName').value,
    device: document.getElementById('device').value,
    conf: parseFloat(document.getElementById('conf').value),
    iou: parseFloat(document.getElementById('iou').value),
    type: document.getElementById('yoloType').value,
    classes: document.getElementById('classes').value.split(',').map(function(c) {
      return c.trim();
    }).filter(function(c) {
      return c.length > 0;
    })
  };
  
  var source_mode = state.sourceMode;
  var rtsp_url = source_mode === 'production' ? document.getElementById('rtspUrl').value : null;
  var videoSourceRaw = source_mode === 'inference' ? document.getElementById('videoSource').value : '';
  var video_files = source_mode === 'inference'
    ? videoSourceRaw.split(/[\n,]/).map(function(v) {
        return v.trim();
      }).filter(function(v) {
        return v.length > 0;
      })
    : null;
  var max_fps = source_mode === 'production' 
    ? parseInt(document.getElementById('maxFpsProduction').value)
    : parseInt(document.getElementById('maxFpsInference').value);
  
  // Time allowed for production mode (kapan AI aktif)
  var time_allowed_start = null;
  var time_allowed_end = null;
  if (source_mode === 'production') {
    var startInput = document.getElementById('timeAllowedStart').value;
    var endInput = document.getElementById('timeAllowedEnd').value;
    time_allowed_start = startInput ? startInput : null;
    time_allowed_end = endInput ? endInput : null;
  }
  
  // Logs configuration
  var logs_base_dir = document.getElementById('logsBaseDir').value || 'logs';
  var enable_log_reading = document.getElementById('enableLogReading').checked;
  var enable_log_deletion = document.getElementById('enableLogDeletion').checked;
  var start_date = document.getElementById('startDate').value || null;
  var end_date = document.getElementById('endDate').value || null;
  
  // Clean lines: remove centroid field from output (store internally but don't send to backend)
  var cleanedLines = state.lines.map(function(line) {
    return {
      id: line.id,
      type: line.type,
      color: line.color,
      coords: line.coords,
      direction: line.direction,
      orientation: line.orientation,
      bidirectional: line.bidirectional
    };
  });
  
  return {
    config_name: configName,
    model: model,
    feature: state.feature || 'region_crowd',
    feature_params: state.featureParams,
    regions: state.regions,
    lines: cleanedLines,
    source_mode: source_mode,
    rtsp_url: rtsp_url,
    video_files: video_files,
    max_fps: max_fps,
    time_allowed_start: time_allowed_start,
    time_allowed_end: time_allowed_end,
    cctv_id: 'office',
    logs_base_dir: logs_base_dir,
    enable_log_reading: enable_log_reading,
    enable_log_deletion: enable_log_deletion,
    start_date: start_date,
    end_date: end_date
  };
}

// ===== LOGS =====
function loadLogs() {
  fetch('/api/logs?count=20')
    .then(function(res) { return res.json(); })
    .then(function(data) {
      var logArea = document.getElementById('logArea');
      logArea.innerHTML = '';
      
      data.logs.forEach(function(log) {
        var entry = document.createElement('div');
        entry.className = 'log-entry';
        
        var time = document.createElement('span');
        time.className = 'log-time';
        time.textContent = log.timestamp + ' ';
        
        var level = document.createElement('span');
        level.className = 'log-level-' + log.level;
        level.textContent = '[' + log.level + '] ';
        
        var msg = document.createElement('span');
        msg.textContent = '[' + log.category + '] ' + log.message;
        
        entry.appendChild(time);
        entry.appendChild(level);
        entry.appendChild(msg);
        
        logArea.appendChild(entry);
      });
      
      // Scroll to bottom
      logArea.scrollTop = logArea.scrollHeight;
    })
    .catch(function(err) {
      console.error('Failed to load logs:', err);
    });
}

function clearLogs() {
  fetch('/api/logs', { method: 'DELETE' })
    .then(function() {
      document.getElementById('logArea').innerHTML = '';
      showSuccess('Logs cleared');
    })
    .catch(function(err) {
      showError('Failed to clear logs: ' + err.message);
    });
}

function log(category, message) {
  console.log('[' + category + '] ' + message);
  
  // Reload logs from server to stay in sync
  setTimeout(function() {
    loadLogs();
  }, 100);
}

// ===== UI HELPERS =====
function showSuccess(message) {
  showStatus(message, 'info');
}

function showError(message) {
  showStatus(message, 'error');
}

function showStatus(message, type) {
  var el = document.getElementById('statusMessage');
  el.textContent = message;
  el.className = 'status-message ' + type + ' show';
  
  setTimeout(function() {
    el.classList.remove('show');
  }, 5000);
}

// ===== START =====
window.onload = function() {
  init();
};
