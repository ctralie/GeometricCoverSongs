/** Programmer: Chris Tralie */

/**
 * A canvas for switching between cover songs along  diagonals 
 * of a cross recurrence plot
 * @param {object} audio_obj - A dictionary of audio parameters, including 
 * 	a handle to the audio widget and a time interval between adjacent rows
 *  of the SSM, and the dimension "dim" of the SSM
 */
function CSMCanvas() {
	// GUI Elements
	this.CSImage = new Image;
	this.canvas = document.getElementById('CrossSimilarityCanvas');
	this.scoreTxt = document.getElementById("score");
	this.csmctx = this.canvas.getContext('2d');
	this.fileInput = document.getElementById('fileInput');
	this.progressBar = new ProgressBar();
	
	// Information about what song is playing
	this.playing = false;
	this.playIdx = 0;
	this.audio_widgets = [new AudioObject(1, [], "Song 1"), new AudioObject(2, [], "Song 2")];
	
	// Variables for handling feature types
	this.selectFeatureType = document.getElementById("FeatureType");
	this.featureType = "";
	this.selectImageType = document.getElementById("ImageType");
	this.imageType = "CSM";
	this.FeatureCSMs = {};
	this.dimsUpdated = false;

	/**
	 * Update the audio, CSM images, and scores after loading in a new dataset
	 */
	this.updateData = function(data) {
		this.pauseAudio();
		this.audio_widgets = [
			new AudioObject(1, data.beats1, data.song1name, data.file1),
			new AudioObject(2, data.beats2, data.song2name, data.file2)
		];
		this.FeatureCSMs = data.FeatureCSMs;

		//Remove all feature types from the last time if they exit;
		for (var i = this.selectFeatureType.options.length - 1; i >= 0; i--) {
			this.selectFeatureType.remove(i);
		}

		for (val in data.FeatureCSMs) {
			var option = document.createElement('option');
			option.text = val;
			this.selectFeatureType.add(option, val);
			this.featureType = val;
		}
		this.featureType = this.selectFeatureType.value; //Display the currently selected feature
		this.imageType = "CSM";
		this.updateSelectedImage();
		this.progressBar.changeToReady();
	}

	/**
	 * Update the CSM image to the currently selected feature type
	 * and display type
	 */
    this.updateSelectedImage = function() {
        if (this.featureType.length > 0) {
            this.CSImage.src = (this.FeatureCSMs[this.featureType])[this.imageType];
            this.dimsUpdated = false;
            requestAnimationFrame(this.updateCSMCanvas.bind(this));
            var score = this.FeatureCSMs[this.featureType].score;
            score = Math.round(score*1000)/1000;
            this.scoreTxt.innerHTML = score;
        }
	}

	/**
	 * Set the currently selected song to play and start the animation loop
	 */
	this.playAudio = function() {
		this.playing = true;
		var audio1 = this.audio_widgets[this.playIdx];
		var audio2 = this.audio_widgets[(this.playIdx+1)%2];
		audio1.play();
		audio2.pause();
		audio1.btidx = 0;
		audio2.btidx = 0;
        requestAnimationFrame(this.updateCSMCanvas.bind(this));
	}

	/**
	 * Pause all audio
	 */
	this.pauseAudio = function() {
		this.playing = false;
		for (var i = 0; i < 2; i++) {
			this.audio_widgets[i].pause();
		}
	}

	/**
	 * Handle clicks on the canvas to jump around in the songs
	 */
	this.releaseClickCSM = function(evt) {
		evt.preventDefault();
		var offsetidxs = [evt.offsetY, evt.offsetX];

		clickType = "LEFT";
		evt.preventDefault();
		if (evt.which) {
			if (evt.which == 3) clickType = "RIGHT";
			if (evt.which == 2) clickType = "MIDDLE";
		}
		else if (evt.button) {
			if (evt.button == 2) clickType = "RIGHT";
			if (evt.button == 4) clickType = "MIDDLE";
		}
		this.playIdx = 0;
		if (evt.button === 1) {
			this.playing = false;
			this.pauseAudio();
			return;
		}
		if (clickType == "RIGHT") {
			this.playIdx = 1;
		}
		this.audio_widgets[this.playIdx].jumpToBeat(offsetidxs[this.playIdx]);
		this.redrawCSMCanvas();
		if (this.playing) {
			this.playAudio();
		}
		return false;
	}

	this.makeClickCSM = function(evt) {
		evt.preventDefault();
		return false;
	}

	this.clickerDraggedCSM = function(evt) {
		evt.preventDefault();
		return false;
	}

	/**
	 * Initialize all mouse/touch listeners
	 */
	this.initCanvasHandlers = function() {
		var canvas = document.getElementById('CrossSimilarityCanvas');
		this.canvas.addEventListener("contextmenu", function(e){ e.stopPropagation(); e.preventDefault(); return false; }); //Need this to disable the menu that pops up on right clicking
		this.canvas.addEventListener('mousedown', this.makeClickCSM.bind(this));
		this.canvas.addEventListener('mouseup', this.releaseClickCSM.bind(this));
		this.canvas.addEventListener('mousemove', this.clickerDraggedCSM.bind(this));

		this.canvas.addEventListener('touchstart', this.makeClickCSM.bind(this));
		this.canvas.addEventListener('touchend', this.releaseClickCSM.bind(this));
		this.canvas.addEventListener('touchmove', this.clickerDraggedCSM.bind(this));

		this.canvas.addEventListener('contextmenu', function dummy(e) { return false });
	}

	/**
	 * Initialize the menu handlers for changing display type and for loading JSON files
	 */
	this.initMenuHandlers = function() {
		this.selectFeatureType.addEventListener('change', function(e){
			this.featureType = e.target.value;
			this.updateSelectedImage();
		}.bind(this));

		this.selectImageType.addEventListener('change', function(e){
			this.imageType = e.target.value;
			this.updateSelectedImage();
		}.bind(this));

		this.fileInput.addEventListener('change', function(e) {
			this.pauseAudio();
			var file = fileInput.files[0];
			var reader = new FileReader();
			reader.onload = function(e) {
				var data = JSON.parse(reader.result);
				this.updateData(data);
			}.bind(this)
			this.progressBar.loading = true;
			this.progressBar.changeLoad();
			reader.readAsText(file);
		}.bind(this));
	}

	/**
	 * Draw the CSM canvas with the lines superimposed
	 */
	this.redrawCSMCanvas = function() {
		if (!this.CSImage.complete) {
			//Keep requesting redraws until the image has actually loaded
			requestAnimationFrame(this.redrawCSMCanvas.bind(this));
		}
		else {
			if (!this.dimsUpdated) {
				this.dimsUpdated = true;
				this.canvas.width = this.CSImage.width;
				this.canvas.height = this.CSImage.height;
			}
			this.csmctx.fillRect(0, 0, this.CSImage.width, this.CSImage.height);
			this.csmctx.drawImage(this.CSImage, 0, 0);
			this.csmctx.beginPath();
			if (this.playIdx == 0) {
				var idx = this.audio_widgets[0].btidx;
				this.csmctx.moveTo(0, idx);
				this.csmctx.lineTo(this.CSImage.width, idx);
			}
			else {
				var idx = this.audio_widgets[1].btidx;
				this.csmctx.moveTo(idx, 0);
				this.csmctx.lineTo(idx, this.CSImage.height);
			}
			this.csmctx.strokeStyle = '#0020ff';
			this.csmctx.stroke();
		}
	}

	/**
	 * Find the nearest beat to where the song is playing and continue
	 * the animation loop
	 */
	this.updateCSMCanvas = function() {
		this.audio_widgets[this.playIdx].updateIdx();
		this.redrawCSMCanvas();
		if (this.playing) {
			requestAnimationFrame(this.updateCSMCanvas.bind(this));
		}
	}

	this.initCanvasHandlers();
	this.initMenuHandlers();
}