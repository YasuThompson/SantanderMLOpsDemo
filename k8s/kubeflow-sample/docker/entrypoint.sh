#!/bin/sh

/usr/bin/sqlcmd config add-endpoint --name m1 --address $1
export SQLCMD_PASSWORD=$2
/usr/bin/sqlcmd config add-user --name m1 --username $3 --password-encryption none
/usr/bin/sqlcmd config add-context --name m1 --endpoint m1 --user m1

/usr/bin/sqlcmd query "${4}" --database $5