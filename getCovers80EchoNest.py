#Programmer: Chris Tralie
#Purpose: To use the EchoNest api to get features for the Covers80 dataset
import requests
import json
import numpy as np
import scipy.io as sio
import os
import pickle

def getFeatures(url, ECHONESTKEY):
    uploadRequestURL = "http://developer.echonest.com/api/v4/track/upload"
    payload = {'format':'json', 'api_key':ECHONESTKEY, 'url':url}
    results = requests.post(uploadRequestURL, payload, timeout = 20)        
    results = results.json()
    if results['response']['status']['code'] == 0:
        trid = results['response']['track']['id']
        results = requests.get("http://developer.echonest.com/api/v4/track/profile?format=json&bucket=audio_summary&api_key=%s&id=%s"%(ECHONESTKEY, trid), timeout = 20)
        results = results.json()
        track = results['response']['track']
        #The audio summary contains a few fields that the analysis url does not.  Namely valence, danceability, speechiness, energy, liveness, acousticness, instrumentalness
        audio_summary = track['audio_summary']
        analysis_url = audio_summary['analysis_url']
        r = requests.get(analysis_url)
        #bars, track, segments, beats, meta, sections, tatums
        results = r.json()
        #Add missing fields from audio_summary
        fields = ['valence', 'danceability', 'speechiness', 'energy', 'liveness', 'acousticness', 'instrumentalness']
        for f in fields:
            results['track'][f] = audio_summary[f]
        return results
    else:
        return None


if __name__ == '__main__':
    #This code assumes the Covers80 dataset has been unextracted
    #and uploaded somewhere.  In this case my site
    baseurl = 'http://people.duke.edu/~cjt16/'
    
    fin = open('EchoNestKey.txt')
    ECHONESTKEY = fin.readlines()[0].rstrip('\n')
    fin.close()
    
    fin = open('covers32k/list1.list', 'r')
    files = fin.readlines()
    fin.close()
    fin = open('covers32k/list2.list', 'r')
    files = files + fin.readlines()
    fin.close()
    files = [f.rstrip() for f in files]
    for f in files:
        print "Doing ", f
        song = "covers32k/%s"%f
        url = baseurl + song + ".mp3"
        outfilename = "%s.txt"%song
        if os.path.isfile(outfilename):
            print "Skipping %s"%url
        else:
            print url
            while True:
                try:
                    results = getFeatures(url, ECHONESTKEY)
                    if not results:
                        print "Error loading ", f
                    else:
                        pickle.dump(results, open(outfilename, "w"))
                        break
                except:
                    print "TRYING AGAIN"
        
