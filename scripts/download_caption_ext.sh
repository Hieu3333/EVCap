FILE_ID="1JwOn-SCyjiLncYnHNMO3WNf0NeCWwgsB"
FILE_NAME="ext_data/caption_ext_memory.pkl"

CONFIRM=$(wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate \
"https://drive.google.com/uc?export=download&id=${FILE_ID}" -O- | \
sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')

wget --load-cookies cookies.txt "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" \
-O ${FILE_NAME}

rm -f cookies.txt


