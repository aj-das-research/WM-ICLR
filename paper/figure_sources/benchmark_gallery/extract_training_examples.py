#!/usr/bin/env python3
"""Extract two TRAIN/frame0 simulation images and one audited phantom view.

Only this provenance preparation needs the original local cache. Figure rendering
uses the resulting public PNGs and never reads raw datasets.
"""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
TASKS={
 'drone':{'manifest_sha256':'94a1f7bdf723a2915d5dc37e3b2353427300c736c3b37bf9030a766f1f9133ed','episode':'drone-train-s53000-d0','sha256':'e68035a0c9b69d292b97bf5a1521edb0381344b2e3ee68d653a14599ddb7b0c8','label':'Drone (simulation)','repository':'https://github.com/learnsyslab/gym-pybullet-drones','revision':'7ebad1ecabd28a7000add2d05f888aa2e837c2cc'},
 'surgery':{'manifest_sha256':'57991e212f345b5e354a6f7ffcf1f5233d861be05af50a3425d955abc51bc8ca','episode':'train-s4200000-d0','sha256':'8a2cb38b1fd2343b03fbf9b377a6b318d447e37e31c2de138c0b928de3ecc048','label':'Tissue manipulation (simulation)','repository':'https://github.com/ScheiklP/sofa_env','revision':'85bf7e05dd088b824794dda0046679df13b13e6e'},
}

def main():
 (HERE/'assets').mkdir(parents=True,exist_ok=True);records=[]
 for task,expected in TASKS.items():
  manifest=ROOT/f'data/extensions/{task}_v1/manifest.json'
  if sha(manifest)!=expected['manifest_sha256']:raise ValueError('Changed manifest: '+task)
  metadata=json.loads(manifest.read_text())
  # Select using training metadata before opening a payload; no outcomes used.
  record=sorted((e for e in metadata['episodes'] if e['split']=='train'),key=lambda e:e['trajectory_id'])[0]
  if record['trajectory_id']!=expected['episode'] or record['sha256']!=expected['sha256']:raise ValueError('Changed deterministic selection')
  source=manifest.parent/record['file']
  if sha(source)!=expected['sha256']:raise ValueError('Changed image payload')
  with np.load(source,allow_pickle=False) as pack:
   images=pack['images'];assert images.dtype==np.uint8 and images.shape==(41,128,128,3)
   frame=images[0].copy()
  destination=HERE/'assets'/f'{task}_{record["trajectory_id"]}_frame0.png'
  Image.fromarray(frame).save(destination)
  assert np.array_equal(np.asarray(Image.open(destination)),frame)
  records.append({'task':task,'label':expected['label'],'medium':'locally recorded simulation RGB','model_scope':'historical context models; not current spatial ShiftWM checkpoints','study_status':'completed development forecasting/planning; final model test pending','split':'train','episode':record['trajectory_id'],'frame_index':0,'selection':'first lexicographic training trajectory, first stored frame; no outcome selection','source_path':str(source.relative_to(ROOT)),'source_sha256':sha(source),'dataset_manifest_path':str(manifest.relative_to(ROOT)),'dataset_manifest_sha256':sha(manifest),'asset':str(destination.relative_to(ROOT)),'asset_sha256':sha(destination),'pixel_sha256':hashlib.sha256(frame.tobytes()).hexdigest(),'shape':list(frame.shape),'crop':'none','resize':'none','pixel_identity_verified':True,'upstream_repository':expected['repository'],'upstream_revision':expected['revision'],'attribution':'Local simulator observation; simulator source is MIT; no real-flight/patient recording or clinical efficacy claim.'})
 audit_path=ROOT/'reports/real_openh_video_audit.json'
 if sha(audit_path)!='63f8298db6d7dea2d63d8c7afac325bf4613c973943e795dbcb191dce16f10ea':raise ValueError('Changed Open-H audit')
 audit=json.loads(audit_path.read_text());video_record=[r for r in audit['downloads'] if r['path'].endswith('.mp4')][0]
 video=ROOT/'data/real_video/openh_sample_v1'/video_record['path']
 if sha(video)!=video_record['sha256']:raise ValueError('Changed audited Open-H video')
 command=['ffmpeg','-v','error','-i',str(video),'-vf','select=eq(n\\,160)','-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
 raw=subprocess.check_output(command,timeout=30);assert len(raw)==480*640*3
 frame=np.frombuffer(raw,dtype=np.uint8).reshape(480,640,3)
 destination=HERE/'assets/openh_episode_000000_frame160.png';Image.fromarray(frame).save(destination)
 assert np.array_equal(np.asarray(Image.open(destination)),frame)
 records.append({'task':'openh','label':'Open-H · input only','medium':'recorded physical endoscopy phantom RGB, not patient surgery','model_scope':'ingestion illustration only; no trained predictor or model evaluation','study_status':'one audited source episode; no performance result','split':'upstream train (no project model split)','episode':'episode_000000','frame_index':160,'selection':'visual-clarity selection from9 uniform source indices0/32/64/96/128/160/192/224/256: first sampled frame with fully visible phantom target and surrounding folds; no model or experimental outcome used','selection_contact_sheet':'paper/design/benchmark_gallery/openh/uniform9_contact_sheet.png','source_path':str(video.relative_to(ROOT)),'source_sha256':sha(video),'audit_path':str(audit_path.relative_to(ROOT)),'audit_sha256':sha(audit_path),'asset':str(destination.relative_to(ROOT)),'asset_sha256':sha(destination),'pixel_sha256':hashlib.sha256(raw).hexdigest(),'shape':list(frame.shape),'crop':'none','resize':'none','pixel_identity_verified':True,'decode':'ffmpeg original resolution RGB24, exact source frame160','decoder_version':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],'source_url':video_record['source_url'],'upstream_revision':audit['source']['revision'],'license':'CC BY 4.0','attribution':'CUHK physical-phantom endoscopy subset of NVIDIA PhysicalAI-Robotics-Open-H-Embodiment; original dataset authors retain credit.'})
 (HERE/'asset_manifest.json').write_text(json.dumps({'schema':'shiftwm_benchmark_gallery_source_images_v1','status':'passed','records':records,'extractor_sha256':sha(__file__),'payload_splits_opened':['train','upstream_train_ingestion_sample'],'validation_test_or_reserved_payloads_opened':0,'new_simulation_or_model_inference':False},indent=2)+'\n')
 print(json.dumps(records,indent=2))

if __name__=='__main__':main()
