/** Programmer: Chris Tralie */

/**
 * An object for holding information about a song
 * @param {int} songidx - Whether this is song 1 or 2
 * @param {list} bts - List of times for beat onsets
 * @param {string} name - Name of the song 
 * @param {string} src - base64 source of audio
 */
function AudioObject(songidx, bts, name, src) {
    this.widget = document.getElementById("audio_widget"+songidx);
	this.bts = bts;
	this.name = name;
	this.widget.src = src; //Initialize audio
	this.btidx = 0; //Keeps track of which beat we're on in this song
	this.songnameTxt = document.getElementById("songname");
    
    /**
     * Update the index of the row/column to be the closest time
     * to the currently playing time in the audio widget
     */
    this.updateIdx = function() {
        //TODO: Make a binary search
        var time = this.widget.currentTime;
        var t = 0;
		var mindiff = Math.abs(time - this.bts[0]);
		for (var i = 1; i < this.bts.length; i++) {
		    var diff = Math.abs(this.bts[i] - time);
		    if (diff < mindiff) {
		        mindiff = diff;
		        t = i;
		    }
		}
		this.btidx = t;
	}
	
	/** Play the audio associated to this widget */
	this.play = function() {
		this.widget.play();
		this.songnameTxt.innerHTML = this.name;
	}

	/** Pause the audio associated to this widget */
	this.pause = function() {
		this.widget.pause();
	}

	/** Jump to a particular beat in the audio
	 * @param{int} btidx - The index of the beat
	 */
	this.jumpToBeat = function(btidx) {
		this.btidx = btidx;
		this.widget.currentTime = this.bts[btidx];
	}
}
