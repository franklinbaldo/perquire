#!/usr/bin/env python3
"""One-shot repository migration to PEP 723 for standalone Python programs."""
from __future__ import annotations
import ast, os, re, sys, tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; SELF=Path(__file__).resolve()
SKIP_PARTS={".git",".venv","venv","src","tests","test","vendor","node_modules","build","dist","site-packages","__pycache__"}
IMPORT_TO_DIST={"bs4":"beautifulsoup4","click":"click","cryptography":"cryptography","cv2":"opencv-python-headless","cyclopts":"cyclopts","duckdb":"duckdb","fastmcp":"fastmcp","fitz":"pymupdf","httpx":"httpx","ibis":"ibis-framework","keyring":"keyring","litellm":"litellm","markdown_it":"markdown-it-py","matplotlib":"matplotlib","mcp":"mcp","mdformat":"mdformat","networkx":"networkx","numpy":"numpy","openai":"openai","anthropic":"anthropic","pandas":"pandas","PIL":"pillow","playwright":"playwright","pyarrow":"pyarrow","pydantic":"pydantic","pydantic_ai":"pydantic-ai","pytest":"pytest","requests":"requests","rich":"rich","ruamel":"ruamel.yaml","secretstorage":"secretstorage","sklearn":"scikit-learn","yaml":"pyyaml"}
UV_PYTHON_SCRIPT=re.compile(r"\buv run(?:\s+--no-sync)?\s+python\s+((?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.py)")
UV_NOSYNC_SCRIPT=re.compile(r"\buv run\s+--no-sync\s+((?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.py)")
def norm_dist(name): return re.sub(r"[-_.]+","-",name).lower()
def requirement_name(req): return norm_dist(re.split(r"[<>=!~;\[\s]",req,maxsplit=1)[0])
def load_project():
 p=ROOT/"pyproject.toml"
 if not p.exists(): return None,">=3.11",{},set()
 d=tomllib.loads(p.read_text()); pr=d.get("project",{}); name=pr.get("name"); rp=pr.get("requires-python",">=3.11"); reqs={}
 for req in pr.get("dependencies",[]): reqs[requirement_name(req)]=req
 for group in pr.get("optional-dependencies",{}).values():
  for req in group: reqs.setdefault(requirement_name(req),req)
 for group in d.get("dependency-groups",{}).values():
  if isinstance(group,list):
   for req in group:
    if isinstance(req,str): reqs.setdefault(requirement_name(req),req)
 local=set()
 for base in (ROOT,ROOT/"src"):
  if base.exists():
   for child in base.iterdir():
    if child.is_dir() and (child/"__init__.py").exists(): local.add(child.name)
 if name: local.add(name.replace("-","_"))
 return name,rp,reqs,local
def imports(tree):
 out=set()
 for n in ast.walk(tree):
  if isinstance(n,ast.Import): out.update(a.name.split(".",1)[0] for a in n.names)
  elif isinstance(n,ast.ImportFrom) and n.level==0 and n.module: out.add(n.module.split(".",1)[0])
 return out
def has_main(tree):
 for n in tree.body:
  if not isinstance(n,ast.If) or not isinstance(n.test,ast.Compare) or len(n.test.ops)!=1 or len(n.test.comparators)!=1 or not isinstance(n.test.ops[0],ast.Eq): continue
  for a,b in ((n.test.left,n.test.comparators[0]),(n.test.comparators[0],n.test.left)):
   if isinstance(a,ast.Name) and a.id=="__name__" and isinstance(b,ast.Constant) and b.value=="__main__": return True
 return False
def candidate(path):
 rel=path.relative_to(ROOT)
 if path.resolve()==SELF or any(x in SKIP_PARTS for x in rel.parts) or path.name=="__init__.py" or path.name.startswith("test_") or path.name.endswith("_test.py"): return None
 text=path.read_text(encoding="utf-8")
 try: tree=ast.parse(text,filename=str(rel))
 except SyntaxError: return None
 first=text.splitlines()[0] if text.splitlines() else ""
 return (tree,text) if ((first.startswith("#!") and "python" in first) or has_main(tree)) else None
def siblings(path): return {p.stem for p in path.parent.glob("*.py") if p.stem!="__init__"}
def map_dep(m,reqs):
 dist=IMPORT_TO_DIST.get(m,m.replace("_","-")); req=reqs.get(norm_dist(dist)); return req or IMPORT_TO_DIST.get(m)
def block(py,deps,pname,ppath):
 lines=["# /// script",f'# requires-python = "{py}"',"# dependencies = [",*[f'#     "{d}",' for d in deps],"# ]"]
 if pname and ppath: lines += ["#","# [tool.uv.sources]",f'# {pname} = {{ path = "{ppath}", editable = true }}']
 return "\n".join(lines+["# ///"])
def insert(text,meta):
 lines=text.splitlines(keepends=True)
 if lines and lines[0].startswith("#!"): lines[0]="#!/usr/bin/env -S uv run --script\n"
 else: lines.insert(0,"#!/usr/bin/env -S uv run --script\n")
 s="".join(lines)
 if "# /// script" in s: return s
 first,sep,rest=s.partition("\n"); return f"{first}\n#\n{meta}\n{rest}" if sep else f"{first}\n#\n{meta}\n"
def main():
 pname,py,reqs,local=load_project(); plans=[]; unknown=[]
 for path in sorted(ROOT.rglob("*.py")):
  found=candidate(path)
  if not found: continue
  tree,text=found; mods=imports(tree); use=bool(mods&local); ext=mods-set(sys.stdlib_module_names)-local-siblings(path); deps=[]
  for m in sorted(ext):
   d=map_dep(m,reqs)
   if d is None: unknown.append(f"{path.relative_to(ROOT)}: unknown import {m!r}")
   else: deps.append(d)
  ppath=None
  if use and pname: deps.insert(0,pname); ppath=Path(os.path.relpath(ROOT,path.parent)).as_posix()
  deps=list(dict.fromkeys(deps)); plans.append((path,insert(text,block(py,deps,pname if use else None,ppath)),deps))
 if unknown:
  print("Refusing ambiguous dependency inference:",file=sys.stderr); [print(f"- {x}",file=sys.stderr) for x in unknown]; return 2
 for path,text,deps in plans:
  ast.parse(text,filename=str(path.relative_to(ROOT))); path.write_text(text,encoding="utf-8",newline="\n"); print(f"PEP 723: {path.relative_to(ROOT)} -> {deps}")
 for path in ROOT.rglob("*"):
  if not path.is_file() or any(x in {".git",".venv","node_modules"} for x in path.parts) or path.resolve()==SELF: continue
  try: old=path.read_text(encoding="utf-8")
  except (UnicodeDecodeError,OSError): continue
  new=UV_NOSYNC_SCRIPT.sub(r"uv run \1",UV_PYTHON_SCRIPT.sub(r"uv run \1",old))
  if new!=old: path.write_text(new,encoding="utf-8",newline="\n"); print(f"call sites: {path.relative_to(ROOT)}")
 return 0
if __name__=="__main__": raise SystemExit(main())
