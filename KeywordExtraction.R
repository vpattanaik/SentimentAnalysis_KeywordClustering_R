# Clear workspace
cat("\014")
graphics.off()
rm(list = ls())

# Add libraries
library(textdata)
library(tidyverse)
library(tidytext)
library(glue)
library(stringr)
library(plotly)
library(NLP)
library(openNLP)
library(tm)
library(knitr)
library(graph)
library(igraph)
library(visNetwork)
library(networkD3)
library(SnowballC)

filePath = ("Datasets/")
fileName = ("fb_cambridge_sample.txt")

############################## PART 1 - Sentiment Analysis ##############################

# stick together the path to the file & file name
sfName <- glue(filePath, fileName, sep = "")
# get rid of any sneaky trailing spaces
sfName <- trimws(sfName)

# read in the new file
fileText <- glue(read_file(sfName))
# remove any dollar signs (they're special characters in R)
fileText <- gsub("\\$", "", fileText) 

# tokenize
tokens <- tibble(text = fileText) %>% unnest_tokens(word, text)

# get BING sentiment from text: 
bing_tokens <- tokens %>%
  inner_join(get_sentiments("bing")) %>% # pull out only sentiment words
  count(sentiment) %>% # count the # of positive & negative words
  spread(sentiment, n, fill = 0) %>% # made data wide rather than narrow
  mutate(sentiment = positive - negative) # # of positive words - # of negative words

# Visualizing
pBing <- plot_ly(
  x = c(names(bing_tokens)),
  y = c(as.numeric(bing_tokens)),
  type = "bar") %>%
  layout(
  title = "Bing Sentiment Value"
  )
pBing

# Word Count
bing_word_counts <- tokens %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE) %>%
  ungroup()

cat("\nBING Word Count:\n")
print(bing_word_counts)

# Visualizing
bing_word_counts %>%
  group_by(sentiment) %>%
  top_n(10) %>%
  ungroup() %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(y = "Contribution to Sentiment (BING)",
       x = NULL) +
  coord_flip()

##############################

# get NRC sentiment from text: 
nrc_tokens <- tokens %>%
  inner_join(get_sentiments("nrc")) %>% # pull out only sentiment words
  count(sentiment) %>% # count the # of positive & negative words
  spread(sentiment, n, fill = 0) %>% # made data wide rather than narrow
  mutate(sentiment = positive - negative) # # of positive words - # of negative words

# Visualizing
pNrc <- plot_ly(
  x = c(names(nrc_tokens)),
  y = c(as.numeric(nrc_tokens)),
  type = "bar") %>%
  layout(
    title = "NRC Sentiment Value"
  )
pNrc

# Word Count
nrc_word_counts <- tokens %>%
  inner_join(get_sentiments("nrc")) %>%
  count(word, sentiment, sort = TRUE) %>%
  ungroup()

cat("\nNRC Word Count:\n")
print(nrc_word_counts)

# Visualizing
nrc_word_counts %>%
  group_by(sentiment) %>%
  top_n(10) %>%
  ungroup() %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(y = "Contribution to Sentiment (NRC)",
       x = NULL) +
  coord_flip()

############################## Initializing & Preprocessing - Ranking ##############################

corp = Corpus(DirSource(filePath, pattern = fileName))

# Text Tokeization
SplitText <- function(Phrase) { 
  unlist(strsplit(Phrase," "))
}

# To check if word is present in word list
IsSelectedWord <- function(Word) {
  ifelse(length(which(selected_words == Word))>0, TRUE, FALSE)
}

# Utility Functions
# POS Tagging
tagPOS <-  function(x, ...) {
  s <- as.String(x)
  word_token_annotator <- Maxent_Word_Token_Annotator() #from openNLP package
  a2 <- NLP::Annotation(1L, "sentence", 1L, nchar(s))  # annotating the sentences in a text.
  a2 <- NLP::annotate(s, word_token_annotator, a2) # annotate each word in a sentence.
  a3 <- NLP::annotate(s, Maxent_POS_Tag_Annotator(), a2)
  a3w <- a3[a3$type == "word"]   #taking annotated "word" 
  POStags <- unlist(lapply(a3w$features, `[[`, "POS"))
  POStagged <- paste(sprintf("%s/%s", s[a3w], POStags), collapse = " ")
  list(POStagged = POStagged, POStags = POStags)
}

# Illustrate usage of tagPOS
#str <- "this is a confrence tutorial session."
#tagged_str <-  tagPOS(str)
#tagged_str

# Select words with given tagID
SelectTaggedWords <- function(Words,tagID) {
  Words[ grep(tagID,Words) ]
}

# Remove Pos Tags 
RemoveTags <- function(Words) {
  sub("/[A-Z]{2,3}","",Words)
}

# To get all words in its window size W
GetWordLinks <- function(position,scope) {
  scope <- ifelse(position+scope>length(words),length(words),position+scope)
  links <- ""
  for (i in (position+1):scope) {
    if ( IsSelectedWord(words[i]) ) links <- c(links,words[i]) # add to word link list if word is present in wordlist-"word"
  }
  
  if (length(links)>1) {
    links[2:length(links)]
  }
  else {
    links <- ""
  }
}

#Contructing Graph-of-Word Representation from Input Text
ConstructTextGraph <- function(n) { 
  word_graph <- new("graphNEL") # graph object (text graph to be constructed)
  i <- 1
  while (i < length(words) ) {   #here "words"- list of tokens after pre-processing
    if ( IsSelectedWord(words[i]) ) {                                   
      links <- GetWordLinks(i,n) # getting words falling within window W for word[i]        
      if (links[1] != "") {                                     
        #cat(i," ",words[i]," - ",paste(c(links),collapse=" "),"\n") # outputting words falling within window size of words[i]
        if ( length(which(nodes(word_graph)==words[i]))==0  ) {     
          word_graph <- addNode(words[i],word_graph)  # if word is not a node in  word_graph then add it.
        }                                               
        # connect this new node to words falling in its co-occurence window.
        for (j in 1:length(links)) {
          if ( length(which(graph::nodes(word_graph)==links[j]))==0 ) {
            word_graph <- addNode(links[j],word_graph)
            word_graph <- addEdge(words[i],links[j],word_graph,1)
          } 
          else { # if words linked to this newly added words are already connected to this word then increment the edge weight.
            if ( (length(which(graph::edges(word_graph,links[j])[[1]]==words[i]))>0 ) ) { 
              prev_edge_weight <- as.numeric(edgeData(word_graph,words[i],links[j],"weight"))
              edgeData(word_graph,words[i],links[j],"weight") <- prev_edge_weight+1
            }
            else {
              word_graph <- addEdge(words[i],links[j],word_graph,1)
            }
          } 
        }
      }
    }
    i <- i+1
  }
  word_graph
}

# Pre-Processing
corp <- tm_map(corp, removeWords, c("'s"))
corp <- tm_map(corp, stripWhitespace) # removing extra spaces (keeping only single space)
corp <- tm_map(corp, tolower) # transforming to lower case
corp <- tm_map(corp, removePunctuation)
words_with_punctuation <- SplitText(as.character(corp[[1]]))

# GRAPH CONSTRUCTION
words <- SplitText(as.character(corp[[1]])) # tokenization
tagged_text <- tagPOS(corp[[1]])
tagged_words <- SplitText(as.character(tagged_text))

# Keep only NN (Nouns) & JJ (Adjectives) tagged words 

tagged_words <-c(SelectTaggedWords(tagged_words,"/NN"),SelectTaggedWords(tagged_words,"/JJ"))
#Remove tags

tagged_words <- RemoveTags(tagged_words)
selected_words <- unique(tagged_words)  # Keeping unique words.      

############################## PART 1.1 - TextRank ##############################

text_graph <- ConstructTextGraph(4)  # co-occurrence of window size 4 (can range from 2-10)

# Visualizing
textIgraph<-igraph.from.graphNEL(text_graph, name = TRUE, weight = TRUE,unlist.attrs = TRUE)
if (any(which_loop(textIgraph) == TRUE)) {
  textIgraph<-simplify(textIgraph,remove.loops=TRUE)
}

# visIgraph(textIgraph) %>%
#   visNodes(size = 35, shape = "circle") %>%
#   visIgraphLayout(layout = "layout_nicely") %>%
#   visOptions(highlightNearest = TRUE, 
#              nodesIdSelection = TRUE) %>%
#   visInteraction(keyboard = TRUE)

# TEXT RANK
# Computing vertex score asin PageRank (igraph package)

keywords_list<- page.rank(textIgraph, algo="prpack", directed=FALSE)$vector # function from igraph package package

# POST-PROCESSING
nodes_num <- length(nodes(text_graph))
keywords_num <- round(nodes_num/3)  # a third of the number of vertices in the graph.

ordered_keywords<- keywords_list[order(keywords_list,decreasing=TRUE)]
final_Keywords<- head(ordered_keywords,keywords_num)

# Finding TextRank of the text graph
textRank <- induced.subgraph(graph=textIgraph,vids=names(final_Keywords))

# Plotting TextRank Graph
# visIgraph(textRank) %>%
#   visNodes(size = 25, shape = "circle") %>%
#   visIgraphLayout(layout = "layout_nicely") %>%
#   visOptions(highlightNearest = TRUE, 
#              nodesIdSelection = TRUE) %>%
#   visInteraction(keyboard = TRUE)

# Keyword List
cat("TextRank Keywords:\n")
print(names(final_Keywords))

trankEdgeList <- as.data.frame(get.edgelist(textRank, names=TRUE))

# Plotting TextRank Graph
pTrank <- simpleNetwork(trankEdgeList,      
  Source = 1,                 # column number of source
  Target = 2,                 # column number of target
  charge = -100,                # numeric value indicating either the strength of the node repulsion (negative value) or attraction (positive value)
  fontSize = 14,               # size of the node names
  fontFamily = "serif",       # font og node names
  linkColour = "#A9CCE3",        # colour of edges, MUST be a common colour for the whole graph
  nodeColour = "#1F618D",     # colour of nodes, MUST be a common colour for the whole graph
  opacity = 1.0,              # opacity of nodes. 0=transparent. 1=no transparency
  zoom = T                    # Can you zoom on the figure?
)
pTrank

############################## PART 1.2 - Kcore Retention ##############################

text_graph_kcore <- ConstructTextGraph(4)  # co-occurrence of window size 4

# Visualizing
textIgraph_kcore<-igraph.from.graphNEL(text_graph_kcore, name = TRUE, weight = TRUE,unlist.attrs = TRUE)
if (any(which_loop(textIgraph_kcore) == TRUE)) {
textIgraph_kcore<-simplify(textIgraph_kcore,remove.loops=TRUE)
}

# visIgraph(textIgraph_kcore) %>%
#   visNodes(size = 25, shape = "circle") %>%
#   visIgraphLayout(layout = "layout_nicely") %>%
#   visOptions(highlightNearest = TRUE, 
#              nodesIdSelection = TRUE) %>%
#   visInteraction(keyboard = TRUE)

# Coreness
coreness <- graph.coreness(textIgraph_kcore)
maxCoreness <- max(coreness)
verticesHavingMaxCoreness <- which(coreness == maxCoreness) 

# Finding Kcore of the text graph
kcore <- induced.subgraph(graph=textIgraph_kcore,vids=verticesHavingMaxCoreness)

# Plotting Kcore Graph
# visIgraph(kcore) %>%
#   visNodes(size = 25, shape = "circle") %>%
#   visIgraphLayout(layout = "layout_nicely") %>%
#   visOptions(highlightNearest = TRUE,
#              nodesIdSelection = TRUE) %>%
#   visInteraction(keyboard = TRUE)

# Keyword List
cat("\nKCore Keywords:\n")
print(names(verticesHavingMaxCoreness))

kcoreEdgeList <- as.data.frame(get.edgelist(kcore, names=TRUE))

# Plotting Kcore Graph
pKcore <- simpleNetwork(kcoreEdgeList,
  Source = 1,                 # column number of source
  Target = 2,                 # column number of target
  charge = -100,                # numeric value indicating either the strength of the node repulsion (negative value) or attraction (positive value)
  fontSize = 14,               # size of the node names
  fontFamily = "serif",       # font og node names
  linkColour = "#A9CCE3",        # colour of edges, MUST be a common colour for the whole graph
  nodeColour = "#1F618D",     # colour of nodes, MUST be a common colour for the whole graph
  opacity = 1.0,              # opacity of nodes. 0=transparent. 1=no transparency
  zoom = T                    # Can you zoom on the figure?
)
pKcore

########################################################################### PART 2 #####

# Calculate node size
lenTR = length(names(final_Keywords))
trSize <- rep(0, lenTR)
for (i in 1: lenTR) {
  trSize[i] = length(grep(names(final_Keywords)[i], words))
}

# Edge list
edgTRank <- character(lenTR)
k = 1
for (i in 1:lenTR) {
  edgTRank[k] = as.String(trankEdgeList$V1[i])
  edgTRank[k + 1] = as.String(trankEdgeList$V2[i])
  k = k + 2
}

# Initialize graph
gpTRank <- graph(c(edgTRank))
plot(gpTRank, vertex.color = c(trSize), vertex.frame.color = "white", vertex.label.dist = 2,
     edge.color	= "gray", edge.width = 2, edge.arrow.size = 0.25, layout = layout_with_graphopt,
     main = "TextRank Graph")

# Calculate adjacency matrix
adjMatTRank <- matrix(0, nrow = lenTR, ncol = lenTR)
colnames(adjMatTRank) <- c(names(final_Keywords))
rownames(adjMatTRank) <- c(names(final_Keywords))

for (i in 1:nrow(trankEdgeList)) {
  adjMatTRank[c(as.String(trankEdgeList[i, 1])), c(as.String(trankEdgeList[i, 2]))] = 1
  adjMatTRank[c(as.String(trankEdgeList[i, 2])), c(as.String(trankEdgeList[i, 1]))] = 1
}

# Visualize heatmap
palf <- colorRampPalette(c("gold", "dark orange")) 
heatmap(adjMatTRank[, lenTR:1], Rowv = NA, Colv = NA, col = palf(100),
        scale = "none", margins = c(10, 10),
        main = "TextRank Heatmap")

##############################

# Calculate node size
lenKC = length(names(verticesHavingMaxCoreness))
kcSize <- rep(0, lenKC)
for (i in 1: lenKC) {
  kcSize[i] = length(grep(names(verticesHavingMaxCoreness)[i], words))
}

# Edge list
edgKCore <- character(lenKC)
k = 1
for (i in 1:lenKC) {
  edgKCore[k] = as.String(kcoreEdgeList$V1[i])
  edgKCore[k + 1] = as.String(kcoreEdgeList$V2[i])
  k = k + 2
}

# Initialize graph
gpKCore <- graph(c(edgKCore))
plot(gpKCore, vertex.color = c(kcSize), vertex.frame.color = "white", vertex.label.dist = 2,
     edge.color	= "gray", edge.width = 2, edge.arrow.size = 0.25, layout = layout_with_graphopt,
     main = "KCore Graph")

# Calculate adjacency matrix
adjMatKCore <- matrix(0, nrow = lenKC, ncol = lenKC)
colnames(adjMatKCore) <- c(names(verticesHavingMaxCoreness))
rownames(adjMatKCore) <- c(names(verticesHavingMaxCoreness))

for (i in 1:nrow(kcoreEdgeList)) {
  adjMatKCore[c(as.String(kcoreEdgeList[i, 1])), c(as.String(kcoreEdgeList[i, 2]))] = 1
  adjMatKCore[c(as.String(kcoreEdgeList[i, 2])), c(as.String(kcoreEdgeList[i, 1]))] = 1
}

# Visualize heatmap
palf <- colorRampPalette(c("gold", "dark orange")) 
heatmap(adjMatKCore[, lenKC:1], Rowv = NA, Colv = NA, col = palf(100),
        scale = "none", margins = c(10, 10),
        main = "KCore Heatmap")

########################################################################### PART 3 #####

# Clustering
cebTRank <- cluster_edge_betweenness(gpTRank)
clpTRank <- cluster_label_prop(gpTRank)
cfgTRank <- cluster_fast_greedy(as.undirected(gpTRank))

# Visulaization
par(mfrow = c(1, 3))
plot(cebTRank, as.undirected(gpTRank), main = "TextRank - Edge Betweenness")
box("figure", col="red4",  lwd = 5)
plot(clpTRank, as.undirected(gpTRank), main = "Propagating Labels")
box("figure", col="green4",  lwd = 5)
plot(cfgTRank, as.undirected(gpTRank), main = "Greedy Optimization")
box("figure", col="blue4",  lwd = 5)

##############################

# Clustering
cebKCore <- cluster_edge_betweenness(gpKCore)
clpKCore <- cluster_label_prop(gpKCore)
cfgKCore <- cluster_fast_greedy(as.undirected(gpKCore))

# Visulaization
par(mfrow = c(1, 3))
plot(cebKCore, as.undirected(gpKCore), main = "KCore - Edge Betweenness")
box("figure", col="red4",  lwd = 5)
plot(clpKCore, as.undirected(gpKCore), main = "Propagating Labels")
box("figure", col="green4",  lwd = 5)
plot(cfgKCore, as.undirected(gpKCore), main = "Greedy Optimization")
box("figure", col="blue4",  lwd = 5)