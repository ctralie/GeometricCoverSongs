//Programmer: Chris Tralie
//Purpose: To provide a canvas for switching between cover songs
//along diagonals of a cross recurrence plot
var CSImage = new Image;
var csmctx;
var offset1 = 0;
var offset1idx = 0;
var offset2 = 0;
var offset2idx = 0;
var playing1 = true;
var playIdxCSM = 0;
var bts1 = [];
var bts2 = [];
var startTime = 0;
var offsetTime = 0;

//Functions to handle mouse motion
function releaseClickCSM(evt) {
	evt.preventDefault();
	offset1idx = evt.offsetY;
	offset2idx = evt.offsetX;

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

	playing1 = true;
	if (evt.button === 1) {
		playing = false;
		source.stop();
		return;
	}
	if (clickType == "RIGHT") {
		playing1 = false;
	}
    if (playing) {
        source.stop();
        if (playing1) {
        	playAudio(1);
        }
        else {
        	playAudio(2);
        }
    }
    else {
    	redrawCSMCanvas();
    }
	return false;
}

function makeClickCSM(evt) {
	evt.preventDefault();
	return false;
}

function clickerDraggedCSM(evt) {
	evt.preventDefault();
	return false;
}

function initCanvasHandlers() {
    var canvas = document.getElementById('CrossSimilarityCanvas');
    canvas.addEventListener("contextmenu", function(e){ e.stopPropagation(); e.preventDefault(); return false; }); //Need this to disable the menu that pops up on right clicking
    canvas.addEventListener('mousedown', makeClickCSM);
    canvas.addEventListener('mouseup', releaseClickCSM);
    canvas.addEventListener('mousemove', clickerDraggedCSM);

    canvas.addEventListener('touchstart', makeClickCSM);
    canvas.addEventListener('touchend', releaseClickCSM);
    canvas.addEventListener('touchmove', clickerDraggedCSM);

    canvas.addEventListener('contextmenu', function dummy(e) { return false });
}

function redrawCSMCanvas() {
	if (!CSImage.complete) {
	    //Keep requesting redraws until the image has actually loaded
	    requestAnimationFrame(redrawCSMCanvas);
	}
	else {
		if (!dimsUpdated) {
			dimsUpdated = true;
			csmcanvas.width = CSImage.width;
            csmcanvas.height = CSImage.height;
		}
        csmctx.fillRect(0, 0, 1000, 1000);
	    csmctx.drawImage(CSImage, 0, 0);
	    csmctx.beginPath();
	    if (playing1) {
		    csmctx.moveTo(0, offset1idx);
		    csmctx.lineTo(CSImage.width, offset1idx);
	    }
	    else {
		    csmctx.moveTo(offset2idx, 0);
		    csmctx.lineTo(offset2idx, CSImage.height);
	    }
	    csmctx.strokeStyle = '#0020ff';
	    csmctx.stroke();
    }
}

function updateCSMCanvas() {
	var t = context.currentTime - startTime + offsetTime;
	var bts;
	if (playing1) {
		bts = bts1;
	}
	else {
		bts = bts2;
	}
	while (bts[playIdxCSM] < t && playIdxCSM < bts.length - 1) {
		playIdxCSM++;
	}
	if (playing1) {
		offset1idx = playIdxCSM;
	}
	else {
		offset2idx = playIdxCSM;
	}
	redrawCSMCanvas();
	if (playing) {
		requestAnimationFrame(updateCSMCanvas);
	}
}
