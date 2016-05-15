print 'Type the filename'
filename=raw_input(">")
print "We`re going to erase %r." %filename
print "Opening the file..."
target=open(filename,'w')
print "Truncating the file."
target.truncate()
print "Write three lines"
line1=raw_input("Write files")
line2=raw_input("Write second file")
line3=raw_input("Write third one")
print "We`re going to write these three files"
target.write(line1)
target.write("\n")
target.write(line2)
target.write("\n")
target.write(line3)
target.write("\n")
print "We`re now closing all files"
target.close()