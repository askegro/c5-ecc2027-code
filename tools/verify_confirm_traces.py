from pathlib import Path
import shutil,tarfile,hashlib,json,collections,gzip
import argparse, tempfile
parser=argparse.ArgumentParser(description='Verify the complete private E2 split archive without extracting data (Python 3.11+).')
parser.add_argument('--parts',type=Path,required=True)
parser.add_argument('--manifest',type=Path,required=True)
parser.add_argument('--report',type=Path,required=True)
parser.add_argument('--work-dir',type=Path,default=Path.cwd(),help='Writable location for temporary joined archive')
args=parser.parse_args()
temp=tempfile.TemporaryDirectory(prefix='confirm-traces-',dir=args.work_dir)
work=Path(temp.name)
parts=sorted(args.parts.glob('traces_confirm.tgz.part*'))
if [p.name for p in parts] != [f'traces_confirm.tgz.part{i:02d}' for i in range(16)]:
 raise SystemExit('Expected exactly part00 through part15')
out=work/'traces_confirm.tgz'
with out.open('wb') as dst:
 for p in parts:
  with p.open('rb') as src:shutil.copyfileobj(src,dst)
with tarfile.open(out,'r:gz') as t:
 members=t.getmembers(); names=[m.name.removeprefix('./') for m in members if m.isfile()]
 print('SAMPLE',names[:8])
expected=[l.split()[0] for l in args.manifest.read_text().splitlines() if l.strip()]
groups=collections.defaultdict(list)
for name in names:
 unit,record=Path(name).name.split('.',1);groups[unit].append(record)
required={'s4786.npz','s9588.npz','short74.npz','passive.npz','val.npz','zref.json'}
# Read through the gzip trailer to verify its CRC and uncompressed length.
with gzip.open(out,'rb') as f:
 while f.read(1024*1024):pass
report={'archive_bytes':out.stat().st_size,'archive_sha256':hashlib.file_digest(out.open('rb'),'sha256').hexdigest(),'part_count':len(parts),'parts':[{'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for p in parts],'regular_files':len(names),'units':len(groups),'manifest_entries':len(expected),'duplicate_archive_paths':[n for n,c in collections.Counter(names).items() if c>1],'duplicate_manifest_paths':[n for n,c in collections.Counter(expected).items() if c>1],'missing':sorted(set(expected)-set(names)),'extra':sorted(set(names)-set(expected)),'invalid_units':{u:r for u,r in groups.items() if set(r)!=required or len(r)!=6},'condition_units':dict(collections.Counter(u.split('_p')[0] for u in groups)),'gzip_integrity':'passed'}
report['passed']=len(names)==576 and len(groups)==96 and len(expected)==576 and not any(report[k] for k in ['duplicate_archive_paths','duplicate_manifest_paths','missing','extra','invalid_units'])
args.report.parent.mkdir(parents=True,exist_ok=True)
args.report.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='parts'},indent=2))
if not report['passed']:
 raise SystemExit('Archive verification failed; see report')

temp.cleanup()
