//A function to show a progress bar
var loading = false;
var loadString = "Loading";
var loadColor = "yellow";
var ndots = 0;
function changeLoad() {
    if (!loading) {
        return;
    }
    var s = "<h3><font color = \"" + loadColor + "\">" + loadString;
    for (var i = 0; i < ndots; i++) {
        s += ".";
    }
    s += "</font></h3>";
    waitingDisp.innerHTML = s;
    if (loading) {
        ndots = (ndots + 1)%12;
        setTimeout(changeLoad, 200);
    }
}
function changeToReady() {
    loading = false;
    waitingDisp.innerHTML = "<h3><font color = \"#00FF00\">Ready</font></h3>";
}
function setLoadingFailed() {
    loading = false;
    waitingDisp.innerHTML = "<h3><font color = \"red\">Loading Failed :(</font></h3>";
}

//Base64 Functions
//http://stackoverflow.com/questions/21797299/convert-base64-string-to-arraybuffer
function base64ToArrayBuffer(base64) {
    var binary =  window.atob(base64);
    var len = binary.length;
    var bytes = new Uint8Array( len );
    for (var i = 0; i < len; i++)        {
        bytes[i] = binary.charCodeAt(i);
    }
    return bytes.buffer;
}

function ArrayBufferTobase64(arrayBuff) {
    var binary = '';
    var bytes = new Uint8Array(arrayBuff);
    var N = bytes.byteLength;
    for (var i = 0; i < N; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
}

//Parameters for doing PCA on music
var MusicParams = {displayTimeEdges:true, needsUpdate:false};
var musicFeatures = {};


function setHeader(xhr) {
    xhr.setRequestHeader('Content-Type', 'text/plain');
}


function loadPrecomputedSong(file) {
    loadString = "Reading data";
    loadColor = "red";
    loading = true;
    changeLoad();

    var xhr = new XMLHttpRequest();
    xhr.open('GET', file, true);
    xhr.responseType = 'json';
    xhr.onload = function(err) {
        processPrecomputedResults(this.response);
    };
    loading = true;
    ndots = 0;
    changeLoad();
    xhr.send();
}


function makeMusicParamsDirty() {
    MusicParams.needsUpdate = true;
    recomputeButton.style.backgroundColor = "red";
    requestAnimFrame(function(){repaintWithContext(context)});
}
