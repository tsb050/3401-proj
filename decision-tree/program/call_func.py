import growTree
import evaluate
import disp
import pruneTree

# Full tree
v1 = growTree.main()
evaluate.main(v1)
disp.showIt(v1)

# Prune tree
v2 = pruneTree.main(v1)
evaluate.main(v2)
