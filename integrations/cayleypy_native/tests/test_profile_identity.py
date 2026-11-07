import zipfile
import pytest
from multigpubeamsearch.profile_identity import torchscript_profile_digest


def archive(path,constants,*,data=b'weights',body='    return self.weight\n',debug=b'debug'):
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('features/data.pkl',b'graph and class bindings')
        z.writestr('features/data/0',data)
        z.writestr('features/code/model.py','class Model:\n'+constants+'  def forward(self, x):\n'+body)
        z.writestr('features/code/model.py.debug_pkl',debug)
        z.writestr('features/.data/serialization_id',debug)


def test_source_maps_archive_id_and_literal_declaration_order_are_portable(tmp_path):
    a=tmp_path/'a.ts';b=tmp_path/'b.ts'
    x='  eps : Final[float] = 1e-5\n';y='  size : Final[int] = 384\n'
    archive(a,x+y);archive(b,y+x,debug=b'another source path')
    assert a.read_bytes()!=b.read_bytes()
    assert torchscript_profile_digest(a)==torchscript_profile_digest(b)


@pytest.mark.parametrize('change',[{'data':b'changed weights'},{'body':'    return x\n'}])
def test_executable_changes_cannot_reuse_profile(tmp_path,change):
    a=tmp_path/'a.ts';b=tmp_path/'b.ts'
    archive(a,'');archive(b,'',**change)
    assert torchscript_profile_digest(a)!=torchscript_profile_digest(b)


def test_nonliteral_declarations_are_not_reordered(tmp_path):
    a=tmp_path/'a.ts';b=tmp_path/'b.ts'
    x='  a : Final[int] = other\n';y='  b : Final[int] = 7\n'
    archive(a,x+y);archive(b,y+x)
    assert torchscript_profile_digest(a)!=torchscript_profile_digest(b)


def test_duplicate_declarations_keep_their_original_order(tmp_path):
    a=tmp_path/'a.ts';b=tmp_path/'b.ts'
    x='  a : Final[int] = 1\n';y='  a : Final[int] = 7\n'
    archive(a,x+y);archive(b,y+x)
    assert torchscript_profile_digest(a)!=torchscript_profile_digest(b)
