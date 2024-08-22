#!/bin/bash

/usr/local/bin/singularity shell $1 <<EOT
python3.6 -u ${@:2}
EOT

exit 0
